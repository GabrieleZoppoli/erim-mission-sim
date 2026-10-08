"""E1: main comparison. Units: train_s<seed> (ERIM training, weights saved), eval_baseline, eval_erim_s<seed>.
Rows go to out/E1/results.csv; stage logs and sample trajectories for the first missions."""
import os
import jax
import numpy as np
from .. import io
from ..erim import agent as eagent
from ..baseline import agent as bagent
from . import common


def units(c):
    return [f"train_s{s}" for s in c.exp.seeds] + ["eval_baseline"] + [f"eval_erim_s{s}" for s in c.exp.seeds]


def weights_path(out, seed):
    return os.path.join(out, "weights", f"erim_s{seed}.npz")


def load_params(c, out, seed):
    cfg, mcfg = c.sim, c.model
    like = eagent.init_params(jax.random.PRNGKey(0), cfg, mcfg, _zero_norm(c))
    return io.load_pytree(weights_path(out, seed), like)


def _zero_norm(c):
    from ..erim.agent import Norm
    from .. import windows
    n_in = (c.model.N + 1) * windows.FEAT
    z = lambda d: np.zeros(d, np.float32)
    o = lambda d: np.ones(d, np.float32)
    return Norm(z(n_in), o(n_in), z(15), o(15), z(6), o(6), z(4), o(4), z(2), o(2))


def run_unit(c, unit, out, log=print):
    cfg = c.sim
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    missions, h = common.eval_missions(c)
    if unit.startswith("train_s"):
        seed = int(unit[len("train_s"):])
        params, mcfg, info = common.train_erim(jax.random.PRNGKey(1000 + seed), c, log=log)
        io.save_pytree(weights_path(out, seed), params)
        io.append_csv(os.path.join(out, "training.csv"), [{"seed": seed, **{k: v for k, v in info.items() if k != "est_losses"},
                                                           **{f"loss_{k}": v for k, v in info["est_losses"].items()}}])
        return info
    if unit == "eval_baseline":
        A = bagent.bind(cfg)
        rows, logs = common.evaluate(A, bagent.make_params(cfg), c, missions, "baseline", -1)
        io.append_csv(os.path.join(out, "results.csv"), rows)
        common.save_stage_logs(out, "baseline", logs, c.exp.stage_log_missions)
        common.save_trajectories(out, "baseline", logs, c.exp.sample_missions, "baseline", -1)
        return {"missions_hash": h, "stopped": float(np.mean([r["stopped"] for r in rows])),
                "class_ok_T": float(np.mean([r["class_ok_T"] for r in rows])), "final_dist": float(np.median([r["final_dist"] for r in rows]))}
    if unit.startswith("eval_erim_s"):
        seed = int(unit[len("eval_erim_s"):])
        params = load_params(c, out, seed)
        A = eagent.bind(cfg, c.model, use_images=True, use_post=False)
        rows, logs = common.evaluate(A, params, c, missions, "erim", seed)
        io.append_csv(os.path.join(out, "results.csv"), rows)
        if seed == c.exp.seeds[0]:
            common.save_stage_logs(out, "erim", logs, c.exp.stage_log_missions)
            common.save_trajectories(out, "erim", logs, c.exp.sample_missions, "erim", seed)
        return {"missions_hash": h, "stopped": float(np.mean([r["stopped"] for r in rows])),
                "class_ok_T": float(np.mean([r["class_ok_T"] for r in rows])), "final_dist": float(np.median([r["final_dist"] for r in rows]))}
    raise ValueError(unit)
