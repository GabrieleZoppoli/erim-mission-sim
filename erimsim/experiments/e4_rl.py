"""E4: PPO from pixels, evaluated on the same missions as E1. Units: ppo_s<seed>.
Two iterates are evaluated per seed: the final parameters (method "ppo", as before) and the parameters of the update
with the highest smoothed training reward (method "ppo_best"; see rl/ppo.py). ppo_best.csv records which update."""
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
    params, curve, best = ppo.train(jax.random.PRNGKey(5000 + seed), cfg, mcfg, c.exp.rl_steps, c.exp.rl_envs, c.exp.rl_unroll, log=log)
    for r in curve:
        r["seed"] = seed
    io.append_csv(os.path.join(out, "learning_curve.csv"), curve)
    io.append_csv(os.path.join(out, "ppo_best.csv"), [{"seed": seed, "best_update": best["update"], "n_updates": best["n_updates"],
                                                       "reward_smooth_best": best["reward_smooth"]}])
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    io.save_pytree(os.path.join(out, "weights", f"ppo_s{seed}.npz"), params)
    io.save_pytree(os.path.join(out, "weights", f"ppo_best_s{seed}.npz"), best["params"])
    log(f"  ppo trained in {time.time() - t0:.0f}s; best smoothed training reward {best['reward_smooth']:.4f} at update {best['update']}/{best['n_updates']}")
    missions, h = common.eval_missions(c)
    A = ppo.bind(cfg)
    rows, logs = common.evaluate(A, params, c, missions, "ppo", seed, chunk=200)
    io.append_csv(os.path.join(out, "results.csv"), rows)
    if seed == c.exp.rl_seeds[0]:
        common.save_trajectories(out, "ppo", logs, c.exp.sample_missions, "ppo", seed)
    rows_b, _ = common.evaluate(A, best["params"], c, missions, "ppo_best", seed, chunk=200)
    io.append_csv(os.path.join(out, "results.csv"), rows_b)
    log(f"  ppo evaluated: final_dist median {np.median([r['final_dist'] for r in rows]):.3f} (final) / {np.median([r['final_dist'] for r in rows_b]):.3f} (best iterate)")
    return {"seed": seed, "final_dist": float(np.median([r["final_dist"] for r in rows])),
            "cost": float(np.mean([r["total_cost"] for r in rows])),
            "final_dist_best": float(np.median([r["final_dist"] for r in rows_b])),
            "cost_best": float(np.mean([r["total_cost"] for r in rows_b])), "best_update": int(best["update"]),
            "train_seconds": time.time() - t0}
