# Run sheet — erim-mission-sim on a two-GPU node

For the operator who runs the study inside the university network. The code never needs to know anything about the
node beyond a working directory and two GPU indices. No placeholders remain.

* Repository: `https://github.com/GabrieleZoppoli/erim-mission-sim` — tag `v0.1`
* Hardware assumed: 2 × NVIDIA RTX A6000 48 GB, driver 550.x, CUDA 12.4; any Python ≥ 3.10.

## 0. Workspace (house rules: personal NVMe workspace, nothing on persistent project storage)

```bash
init_workspace PROJ_ERIM_SIM other simulation          # once; creates /data02/work/<user>/PROJ_ERIM_SIM/other/simulation/{dev/code,tmp,res,...}
cd /data02/work/<user>/PROJ_ERIM_SIM/other/simulation/dev/code
git clone https://github.com/GabrieleZoppoli/erim-mission-sim && cd erim-mission-sim && git checkout v0.1
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
| GPU0: E3 | 9 (3 λ × 3 seeds) | ≈ 40 min | ≈ 6 h | ≤ 20 GB |
| GPU0: E2 CNN + ref | 18 + 1 | ≈ 8 min | ≈ 2.5 h | ≤ 10 GB |
| GPU1: E2 estimators | 36 units × 3 seeds, two shards | ≈ 5 min per seed | ≈ 4.5 h wall-clock | 2 × 45 % |
| GPU1: E4 | 3 seeds | ≈ 3 h | ≈ 9 h | ≤ 16 GB |

Budget: about 30 GPU-hours planned against the 48 agreed. If a stream runs long, cut in this order: E4 seeds 3→1
(`--seeds 0`), E2 estimator grid thinned (`--units 'est_n*_M50' ...`), E3 seeds 3→2, E1 seeds 5→3, evaluation missions
1000→500 (edit `M_eval` in `config.py`).

## 4. What to return (small files only; weights are optional)

```bash
cd $OUT && tar czf erimsim_results_$(date +%Y%m%d).tgz --exclude='weights' E1 E2 E3 E4 bench_report.txt gpu0.log gpu1.log
du -sh erimsim_results_*.tgz          # expected 20–60 MB (stage logs are float16, 50 missions)
tar czf erimsim_weights_$(date +%Y%m%d).tgz E1/weights E3/weights E4/weights    # ≈ 30 MB, send if convenient
```
Contents: `E*/results.csv`, `E1/training.csv`, `E2/rates_*.csv`, `E4/learning_curve.csv`, `E*/trajectories_sample.csv`,
`E*/stage_logs_*.npz`, `E*/meta.json`, `E*/log_*.txt`. No file outside `$OUT` is created; the workspace can be deleted
afterwards.
