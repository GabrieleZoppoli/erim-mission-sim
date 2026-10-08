"""E3: dual effect. The policy sees the class posterior and the cost carries lambda * entropy(posterior); the
renderer and the recogniser are inside the policy-search loop. Warm start from the E1 weights of the same seed when
available (out/E1/weights), otherwise a fresh training. Units: dual_l<lambda>_s<seed>."""
import os
import time
import numpy as np
import jax
from .. import io
from ..erim import agent as eagent, policy
from . import common, e1_main


def units(c):
    return [f"dual_l{l:g}_s{s}" for l in c.exp.lambdas for s in c.exp.e3_seeds]


def run_unit(c, unit, out, log=print):
    cfg, mcfg, tcfg = c.sim, c.model, c.train
    lam = float(unit.split("_")[1][1:]); seed = int(unit.split("_")[2][1:])
    e1_out = os.path.join(os.path.dirname(out), "E1")
    missions, h = common.eval_missions(c)
    t0 = time.time()
    if os.path.exists(e1_main.weights_path(e1_out, seed)):
        params = e1_main.load_params(c, e1_out, seed)
        log(f"  warm start from E1 seed {seed}")
    else:
        params, _, _ = common.train_erim(jax.random.PRNGKey(1000 + seed), c, log=log)
    A = eagent.bind(cfg, mcfg, use_images=True, use_post=True)
    params, info = policy.train_policy(jax.random.PRNGKey(3000 + seed), params, A, cfg, tcfg, lam=lam, with_images=True,
                                       steps=c.exp.e3_pol_steps, batch=c.exp.e3_batch, restarts=1)
    log(f"  dual policy trained (lambda={lam}): loss {info['best_loss']:.3f} ({time.time() - t0:.0f}s)")
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    io.save_pytree(os.path.join(out, "weights", f"dual_l{lam:g}_s{seed}.npz"), params)
    rows, logs = common.evaluate(A, params, c, missions, f"dual_l{lam:g}", seed, chunk=200, lam=lam)
    for r in rows:
        r["lambda"] = lam
    io.append_csv(os.path.join(out, "results.csv"), rows)
    if seed == c.exp.e3_seeds[0]:
        common.save_trajectories(out, f"dual_l{lam:g}", logs, c.exp.sample_missions, f"dual_l{lam:g}", seed)
        common.save_stage_logs(out, f"dual_l{lam:g}", logs, c.exp.stage_log_missions)
    return {"lambda": lam, "seed": seed, "class_ok_T": float(np.mean([r["class_ok_T"] for r in rows])),
            "t_first_correct": float(np.median([r["t_first_correct"] for r in rows])),
            "final_dist": float(np.median([r["final_dist"] for r in rows])), "cost": float(np.mean([r["total_cost"] for r in rows]))}
