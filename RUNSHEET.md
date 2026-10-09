# Run sheet — erim-mission-sim on a two-GPU node

For the operator who runs the study inside the university network. The code never needs to know anything about the
node beyond a working directory and two GPU indices. No placeholders remain.

* Repository: `https://github.com/GabrieleZoppoli/erim-mission-sim` — commit `b727351` (the code of tag `v0.1`; the tag itself may not be on GitHub yet)
* Hardware assumed: 2 × NVIDIA RTX A6000 48 GB, driver 550.x, CUDA 12.4; Python 3.11–3.13 (jax 0.10.2 requires ≥ 3.11: if the system Python is older, build the venv from any CPython 3.11–3.13 present on the node, e.g. `/path/to/python3.13 -m venv .venv`).

## 0. Workspace (house rules: personal NVMe workspace, nothing on persistent project storage)

```bash
init_workspace PROJ_ERIM_SIM other simulation          # once; creates /data02/work/<user>/PROJ_ERIM_SIM/other/simulation/{dev/code,tmp,res,...}
cd /data02/work/<user>/PROJ_ERIM_SIM/other/simulation/dev/code
git clone https://github.com/GabrieleZoppoli/erim-mission-sim && cd erim-mission-sim && git checkout b727351
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
pip install -r requirements.txt && pip install "jax[cuda12]==0.10.2"      # route A; see README for routes B and C
python3 -c "import jax; print(jax.devices())"                              # expect two CudaDevice entries
export OUT=/data02/work/<user>/PROJ_ERIM_SIM/other/simulation/res/erimsim   # all outputs go here
```

## 1. Bench first (about 5 minutes) — send `bench_report.txt` back before launching anything long

```bash
bash scripts/bench.sh $OUT 0
```
It prints `nvidia-smi`, the Python and JAX versions, runs the four experiments at smoke size, and times the building
blocks at full size. The timings below are extrapolated from a CPU; the bench report is what confirms them.

## 2. Launch (two independent streams, each in its own nohup; tmux is fine too)

```bash
nohup bash scripts/run_gpu0.sh $OUT > $OUT/gpu0.log 2>&1 &     # E1 -> E3 -> E2 CNN sweep + reference
nohup bash scripts/run_gpu1.sh $OUT > $OUT/gpu1.log 2>&1 &     # E2 estimator sweep as 2 shards (45% memory each) -> E4
```
Monitoring: `tail -f $OUT/gpu0.log $OUT/gpu1.log`; completed units appear as markers in `$OUT/E*/units/`. Every
command is restartable with the same line (`--resume` is set in the scripts). Kill with `pkill -f "erimsim --exp"`.

## 3. Expected duration and memory (to be confirmed by the bench)

| Stream | Units | Per unit | Total | GPU memory |
|---|---|---|---|---|
| GPU0: E1 | 5 trainings + 6 evaluations | train ≈ 20–30 min, eval ≈ 3 min | ≈ 2.5 h | ≤ 12 GB |
| GPU0: E3 | 9 (3 λ × 3 seeds) | ≈ 1.5–2 h (from the bench: images-in-loop gradient ≈ 3 × the 1.2 s forward per 256 × 100, 2000 steps, horizons 50–300) | ≈ 15–18 h | ≤ 10 GB after the per-stage rematerialisation of 8 Oct |
| GPU0: E2 CNN + ref | 18 + 1 | ≈ 8 min | ≈ 2.5 h | ≤ 10 GB |
| GPU1: E2 estimators | 36 units × 3 seeds, two shards | ≈ 5 min per seed | ≈ 4.5 h wall-clock | 2 × 45 % |
| GPU1: E4 | 3 seeds | ≈ 3 h | ≈ 9 h | ≤ 16 GB |

Budget: about 35 GPU-hours planned against the 48 agreed (E3 is the long pole). If a stream runs long, cut in this
order: E3 seeds 3→2 (`--seeds 0,1`), E4 seeds 3→1 (`--seeds 0`), E2 estimator grid thinned (`--units 'est_n*_M50' ...`),
E1 seeds 5→3, evaluation missions 1000→500 (edit `M_eval` in `config.py`).

Note of 8 Oct: the first bench on the node showed the E3 gradient asking for 146 GB at batch 128 × 100 stages; commit
`3f278e0` rematerialises the stage inside the rollout when images are in the loop (E1, E2 and E4 code paths untouched).
If the streams were launched from an earlier commit, do not check out the new one in the running clone: clone it into a
sibling directory and run the E3 line from there with the same `--out` once E1 has finished (E3 warm-starts from
`E1/weights`).

## 3b. Rerun after the policy-search fix of 8 Oct (commit `d5b2976`)

The first full-size run showed the ERIM policy search diverging to a non-finite loss late in training (all E1 seeds,
one E2 seed in three); the code then evaluated the untrained initial policy. From commit `d5b2976` every step is
checked (non-finite steps are skipped and halve the learning rate) and the policy returned is the best finite
validation iterate; the log carries a line every val_every steps. Void outputs: E1 (ERIM rows, weights), E2
`rates_est.csv` and `rates_ref.csv`, E3. Still valid: E1 baseline rows (cheap to regenerate), E2 `rates_cnn.csv`, E4.

```bash
git clone https://github.com/GabrieleZoppoli/erim-mission-sim erim-mission-sim-fix && cd erim-mission-sim-fix && git checkout d5b2976
. ../erim-mission-sim/.venv/bin/activate            # the same environment
pkill -f "shard 0/2"; pkill -f "shard 1/2"          # the void E2 estimator shards; the GPU1 wrapper then goes on to E4
bash scripts/rerun_after_fix.sh $OUT                 # sets the void results aside, keeps cnn_* and E4
nohup bash scripts/run_gpu0.sh $OUT > $OUT/gpu0_rerun.log 2>&1 &     # when the cnn_* sweep of the first run has finished
nohup bash scripts/run_gpu1.sh $OUT > $OUT/gpu1_rerun.log 2>&1 &     # when E4 has finished (or now with --mem-fraction 0.3 if memory allows)
```
A training unit now prints `policy trained: best validation loss ... (initial ..., restart r, step k, n skipped steps)`
and `training.csv` carries `pol_improved`; a unit with `pol_improved = False` is a failure to report, not a result.

## 3c. E4 best-iterate evaluation (9 Oct, commit after `9af9f53`)

During the second E4 run the return-scale estimate jumped twentyfold around update 450 (a few saturation episodes
in one batch). The training algorithm is unchanged; `ppo.train` now also keeps the parameters of the update with the
highest smoothed training reward (EMA over about ten updates), outside the jitted update, so a given seed follows the
same trajectory as before. E4 evaluates both iterates on the same 1000 missions: method `ppo` (final parameters, as
before) and `ppo_best`; `ppo_best.csv` records the chosen update and `weights/ppo_best_s<seed>.npz` the parameters.
Use the new commit for every E4 unit started from now on; a seed already finished with `9af9f53` has `ppo` rows only
and can be rerun later for its `ppo_best` rows if the budget allows (about 3.5 h per seed).

## 3d. E3 post-processing: control cost without the information term (9 Oct)

`E3/results.csv` reports `total_cost` including the information term λ·Σ_t entropy(posterior_t), because the
evaluation uses the same cost as the training. To separate control cost from information term, re-evaluate the nine
dual policies with λ = 0 on the same missions after the GPU0 chain has finished (about 40 s per policy on an A6000):

```bash
python3 -m erimsim.experiments.e3_reeval --out $OUT --gpu 0 > $OUT/e3_reeval.log 2>&1
```
It writes `E3/results_control_cost.csv` (method `dual_l<λ>_c0`, plus `lambda`, `entropy_sum`, `entropy_mean`,
`entropy_T`) and one done marker per policy under `E3/units/`; rerunnable. The log line per unit prints
`control cost`, `entropy_sum` and `implied total`, which must equal the unit's cost in the E3 log. Return the CSV
and the log with the rest.

## 4. What to return (small files only; weights are optional)

```bash
cd $OUT && tar czf erimsim_results_$(date +%Y%m%d).tgz --exclude='weights' E1 E2 E3 E4 bench_report.txt gpu0.log gpu1.log
du -sh erimsim_results_*.tgz          # expected 20–60 MB (stage logs are float16, 50 missions)
tar czf erimsim_weights_$(date +%Y%m%d).tgz E1/weights E3/weights E4/weights    # ≈ 30 MB, send if convenient
```
Contents: `E*/results.csv`, `E1/training.csv`, `E2/rates_*.csv`, `E4/learning_curve.csv`, `E*/trajectories_sample.csv`,
`E*/stage_logs_*.npz`, `E*/meta.json`, `E*/log_*.txt`. No file outside `$OUT` is created; the workspace can be deleted
afterwards.
