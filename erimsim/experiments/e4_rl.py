"""E4: PPO from pixels, evaluated on the same missions as E1. Units: ppo_s<seed>."""
import os
import time
import numpy as np
import jax
from .. import io
from ..rl import ppo
from . import common


def units(c):
    return [f"ppo_s{s}" for s in c.exp.rl_seeds]


def run_unit(c, unit, out, log=print):
    cfg, mcfg = c.sim, c.model
    seed = int(unit[len("ppo_s"):])
    t0 = time.time()
    params, curve = ppo.train(jax.random.PRNGKey(5000 + seed), cfg, mcfg, c.exp.rl_steps, c.exp.rl_envs, c.exp.rl_unroll, log=log)
    for r in curve:
        r["seed"] = seed
    io.append_csv(os.path.join(out, "learning_curve.csv"), curve)
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    io.save_pytree(os.path.join(out, "weights", f"ppo_s{seed}.npz"), params)
    log(f"  ppo trained in {time.time() - t0:.0f}s")
    missions, h = common.eval_missions(c)
    A = ppo.bind(cfg)
    rows, logs = common.evaluate(A, params, c, missions, "ppo", seed, chunk=200)
    io.append_csv(os.path.join(out, "results.csv"), rows)
    if seed == c.exp.rl_seeds[0]:
        common.save_trajectories(out, "ppo", logs, c.exp.sample_missions, "ppo", seed)
    return {"seed": seed, "final_dist": float(np.median([r["final_dist"] for r in rows])),
            "cost": float(np.mean([r["total_cost"] for r in rows])), "train_seconds": time.time() - t0}
