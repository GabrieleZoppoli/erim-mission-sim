# erim-mission-sim

Simulation study for a Station–Machine–Object mission problem under imperfect information: a rigid-body Machine
must reach a drifting Object whose dynamics and sensor parameters are unknown, while estimating its own state,
the Object's state, recognising the Object from images and identifying the unknown parameters, with a stopping
procedure based on online quality proxies. Every decision function is realised twice:

* **ERIM** (Extended Ritz Method): fixed-structure parametrised functions, trained by stochastic gradient on
  sampled missions. Ensembles of MLP estimators on a moving window (P_x, P_z, P_f, P_g), a convolutional
  recogniser with Bayesian accumulation (P_X), an MLP policy obtained by direct policy search through the
  dynamics (C_t).
* **Classical chain**: error-state Kalman filter for the Machine, extended Kalman filter with augmented
  parameters for the Object, silhouette template recogniser, LQ control under certainty equivalence.

Everything simulation-side is JAX (`vmap` over missions, `lax.scan` over stages), so the same code runs on CPU
for tests and on CUDA GPUs for the study. R scripts do the statistics and figures from the CSV outputs.
The code is prepared for R. Zoppoli's work on neural approximations for optimal control; it contains no text of
that manuscript.

## The problem (defaults in `erimsim/config.py`)

| | |
|---|---|
| Machine | position, attitude (quaternion), velocity, angular velocity (13 numbers); bounded body-frame accelerations; dt = 0.1 s, 300 stages |
| Object | drifting rigid body, world-frame velocity rotating about an unknown angular velocity (3) with unknown drag (1) = **f**; range sensor with unknown bias and scale = **g** |
| Observations | noisy own state; range / azimuth / elevation of the Object with noise growing with range; 64×64 image of the Object, one of 8 solids rendered by ray-marching signed distance functions, random light, blur and noise growing with range |
| Cost | Σ (distance² + 0.1 ‖u‖²) + terminal distance² ; E3 adds λ·entropy(class posterior) |
| Stopping | Procedure EQF(t): all five proxies below thresholds for T_cons consecutive stages after a trial stage, applied post hoc to logged proxies over a grid of thresholds |

## Experiments

| | What | Output |
|---|---|---|
| **E1** | ERIM vs classical on 1000 identical missions (common random numbers), 5 training seeds, threshold sensitivity | `E1/results.csv` (one row per mission and method), `training.csv`, `trajectories_sample.csv`, `stage_logs_*.npz`, `weights/` |
| **E2** | rate check: one-hidden-layer estimators of width n trained on M missions; CNN of depth J; held-out and training cost, errors, generalisation gap; high-capacity reference | `E2/rates_est.csv`, `rates_cnn.csv`, `rates_ref.csv` |
| **E3** | dual effect: policy sees the class posterior, cost + λ·entropy, renderer and recogniser inside policy search; λ ∈ {0, 0.1, 1} | `E3/results.csv`, `trajectories_sample.csv` |
| **E4** | PPO from pixels and proprioception, same missions | `E4/results.csv`, `learning_curve.csv` |

## Install

Python ≥ 3.10. Three routes, in order of preference.

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
pip install -r requirements.txt                 # CPU build of JAX: tests, smoke, --mini runs
# GPU, route A (pip wheels bundle the CUDA libraries; needs NVIDIA driver >= 525):
pip install "jax[cuda12]==0.10.2"
# GPU, route B (use the CUDA toolkit installed on the node, CUDA 12.x):
pip install "jax[cuda12_local]==0.10.2"
# Route C: if no CUDA wheel installs, keep the CPU build and run with --mini sizes.
python3 -c "import jax; print(jax.devices())"   # must list CudaDevice(s) for routes A/B
```

## Run

```bash
python3 -m pytest -q tests                      # ~3 minutes on 4 CPU cores, 29 tests
python3 -m erimsim --exp E1 --out out --smoke   # 90 s end to end, tiny sizes
python3 -m erimsim --exp E1 --out out --mini    # ~30 min on CPU, intermediate sizes
bash scripts/bench.sh out 0                     # environment report + all smoke tests + benchmark on GPU 0
nohup bash scripts/run_gpu0.sh out > out/gpu0.log 2>&1 &    # E1 -> E3 -> E2 CNN + reference
nohup bash scripts/run_gpu1.sh out > out/gpu1.log 2>&1 &    # E2 estimator sweep (2 shards) -> E4
python3 -m erimsim --exp E2 --out out --list    # list the units of an experiment
Rscript R/e1_compare.R out; Rscript R/e1_thresholds.R out; Rscript R/e2_slopes.R out; Rscript R/e3_dual.R out; Rscript R/e4_curves.R out
```

Runner options: `--gpu N` (sets `CUDA_VISIBLE_DEVICES`), `--mem-fraction 0.45` (two processes per GPU),
`--resume` (skip completed units, marked in `out/<exp>/units/`), `--shard k/n`, `--units GLOB`, `--seeds 0,1`.
Every unit appends its rows to CSV files and leaves a marker, so an interrupted run resumes where it stopped.
**Nothing is written outside `--out`** (and the virtual environment). Memory is not preallocated.

## Layout

```
erimsim/   config.py quat.py dynamics3d.py sensors.py render.py models.py windows.py sim.py procedure.py io.py runner.py bench.py
           erim/ (agent, estimators, recogniser, policy)   baseline/ (ekf, template, lq, agent)   rl/ (ppo)
           experiments/ (common, e1_main, e2_rates, e3_dual, e4_rl)
R/         common.R e1_compare.R e1_thresholds.R e2_slopes.R e3_dual.R e4_curves.R
tests/     29 tests: quaternions, dynamics, sensors, renderer (incl. analytic gradient), models, windows, io, procedure,
           rollout and common random numbers, baseline EKF consistency, ERIM pipeline, runner smoke
scripts/   bench.sh run_gpu0.sh run_gpu1.sh
```

Design notes: the window and target definitions live in one module and serve both training and online use; the
renderer's backward pass uses the implicit function theorem at the hit point (exact, cheap, finite); both agents
draw their noise from `fold_in(mission.key, t)`, so they face identical realisations; the evaluation missions come
from a fixed seed and their hash is recorded in every log.

## Licence

MIT, see `LICENSE`.
