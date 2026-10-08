"""Configuration dataclasses. Everything that sizes the problem or the training lives here, so that an experiment is
a config plus a seed. `smoke()` returns the tiny settings used by the CPU smoke test."""
from dataclasses import dataclass, replace, asdict, field
import json


@dataclass(frozen=True)
class SimCfg:
    dt: float = 0.1
    T: int = 300                 # stages per mission
    img: int = 64                # image side (pixels)
    n_classes: int = 8
    u_max: float = 2.0           # |body acceleration| per axis, m/s^2
    w_max: float = 1.0           # |body angular acceleration| per axis, rad/s^2
    # process noise (sd per step)
    sd_v: float = 0.02
    sd_w: float = 0.01
    sd_vz: float = 0.02
    # own-state observation noise (sd)
    sd_yp: float = 0.05
    sd_yq: float = 0.01          # attitude, rad
    sd_yv: float = 0.02
    sd_yw: float = 0.01
    # Object sensor: range / azimuth / elevation, sd grows as (1 + r/r0)
    sd_r0: float = 0.2
    sd_a0: float = 0.03
    r0: float = 10.0
    # initial conditions and unknown parameters
    r_min: float = 8.0
    r_max: float = 12.0
    sd_vz0: float = 0.3          # Object initial speed per axis
    om_max: float = 0.4          # Object angular velocity (world frame), per axis, rad/s
    cd_max: float = 0.2          # Object drag coefficient
    b_max: float = 0.5           # range bias
    s_dev: float = 0.1           # range scale in [1-s_dev, 1+s_dev]
    # image formation
    k_focal: float = 2.0         # apparent radius = k_focal * img / range  (object radius 1 m)
    march_steps: int = 32
    blur0: float = 0.3           # blur sd (px) = blur0 + blur_r * r
    blur_r: float = 0.08
    sd_px0: float = 0.03         # pixel noise sd = sd_px0 + sd_px_r * r / r0
    sd_px_r: float = 0.05
    # cost
    c_u: float = 0.1
    c_term: float = 10.0


@dataclass(frozen=True)
class ModelCfg:
    N: int = 10                  # window: last N+1 stages
    n_hidden: int = 128
    depth: int = 2
    ensemble: int = 5
    cnn_J: int = 4
    cnn_ch: tuple = (16, 32, 64, 64)
    cnn_fc: int = 128
    pol_hidden: int = 64


@dataclass(frozen=True)
class TrainCfg:
    n_train_missions: int = 400
    T_train: int = 150
    est_steps: int = 20000
    est_bs: int = 256
    est_lr: float = 1e-3
    cnn_steps: int = 30000
    cnn_bs: int = 256
    cnn_lr: float = 1e-3
    pol_steps: int = 3000
    pol_batch: int = 256
    pol_H: int = 300
    pol_lr: float = 1e-3
    pol_restarts: int = 3
    pol_curriculum_start: int = 50


@dataclass(frozen=True)
class ExpCfg:
    name: str = "E1"
    M_eval: int = 1000
    seeds: tuple = (0, 1, 2, 3, 4)
    eps_scales: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    T_cons: tuple = (4, 8, 16)
    t_hat: int = 20
    lambdas: tuple = (0.0, 0.1, 1.0)
    n_grid: tuple = (8, 16, 32, 64, 128, 256)
    M_grid: tuple = (50, 100, 200, 400, 800, 1600)
    J_grid: tuple = (2, 3, 4)
    n_ref: int = 512
    M_ref: int = 3200
    J_ref: int = 5
    rl_steps: int = 50_000_000
    rl_seeds: tuple = (0, 1, 2)
    rl_envs: int = 1024
    rl_unroll: int = 32
    sample_missions: int = 20
    stage_log_missions: int = 50
    # E2 (rate sweep): smaller trainings, one-hidden-layer single networks as in the theorem
    e2_seeds: tuple = (0, 1, 2)
    e2_est_steps: int = 8000
    e2_pol_steps: int = 600
    e2_cnn_steps: int = 5000
    e2_M_test: int = 200
    # E3 (dual effect): policy search with the renderer and the recogniser in the loop
    e3_pol_steps: int = 2000
    e3_batch: int = 128
    e3_seeds: tuple = (0, 1, 2)


@dataclass(frozen=True)
class Cfg:
    sim: SimCfg = field(default_factory=SimCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    exp: ExpCfg = field(default_factory=ExpCfg)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


def preset(name: str) -> Cfg:
    c = Cfg(exp=ExpCfg(name=name))
    return c


def smoke(c: Cfg) -> Cfg:
    """Tiny sizes so that every experiment runs end to end on a CPU in well under two minutes."""
    return Cfg(
        sim=replace(c.sim, T=40, img=16, march_steps=16),
        model=replace(c.model, N=5, n_hidden=32, ensemble=2, cnn_J=2, cnn_ch=(8, 8), cnn_fc=16, pol_hidden=16),
        train=replace(c.train, n_train_missions=16, T_train=40, est_steps=30, est_bs=64, cnn_steps=30, cnn_bs=32,
                      pol_steps=10, pol_batch=8, pol_H=20, pol_restarts=1, pol_curriculum_start=10),
        exp=replace(c.exp, M_eval=8, seeds=(0,), eps_scales=(0.5, 1.0), T_cons=(4,), t_hat=5, lambdas=(0.0, 1.0),
                    n_grid=(8, 16), M_grid=(8, 16), J_grid=(2,), n_ref=16, M_ref=16, J_ref=2, rl_steps=512, rl_seeds=(0,),
                    rl_envs=8, rl_unroll=8, sample_missions=2, stage_log_missions=2,
                    e2_seeds=(0,), e2_est_steps=20, e2_pol_steps=5, e2_cnn_steps=20, e2_M_test=8,
                    e3_pol_steps=4, e3_batch=4, e3_seeds=(0,)),
    )


def mini(c: Cfg) -> Cfg:
    """Intermediate sizes for a CPU sanity run (tens of minutes): enough training to see the behaviour."""
    return Cfg(
        sim=replace(c.sim, T=150, img=32, march_steps=24),
        model=replace(c.model, n_hidden=64, ensemble=3, cnn_J=3, cnn_ch=(8, 16, 16), cnn_fc=32),
        train=replace(c.train, n_train_missions=60, T_train=100, est_steps=2000, cnn_steps=1500, cnn_bs=64,
                      pol_steps=300, pol_batch=64, pol_H=150, pol_restarts=1, pol_curriculum_start=50),
        exp=replace(c.exp, M_eval=40, seeds=(0,), rl_steps=200_000, rl_seeds=(0,), rl_envs=64, rl_unroll=32,
                    sample_missions=5, stage_log_missions=10, e2_seeds=(0,), e2_est_steps=1500, e2_pol_steps=150,
                    e2_cnn_steps=800, e2_M_test=40, n_grid=(8, 32, 128), M_grid=(20, 60), J_grid=(2, 3),
                    n_ref=128, M_ref=60, J_ref=3, e3_pol_steps=150, e3_batch=32, e3_seeds=(0,)),
    )
