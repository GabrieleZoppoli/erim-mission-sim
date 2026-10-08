"""Shared pieces of the experiments: evaluation missions, training of a full ERIM parameter set, evaluation of an
agent into per-mission rows (with the Procedure EQF(t) grid applied post hoc), stage logs and sample trajectories."""
import os
import time
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from .. import sim, dynamics3d as dyn, procedure, io
from ..erim import agent as eagent, estimators, recogniser, policy
from ..baseline import agent as bagent

EVAL_SEED = 12345


def eval_missions(c):
    ms = dyn.sample_missions(jax.random.PRNGKey(EVAL_SEED), c.exp.M_eval, c.sim)
    h = hashlib.sha1(np.asarray(ms.x0).tobytes() + np.asarray(ms.z0).tobytes()).hexdigest()[:12]
    return ms, h


def train_erim(key, c, n_hidden=None, cnn_J=None, n_missions=None, ensemble=None, pol_steps=None, log=print):
    """Full ERIM training for one seed: exploratory data, estimators, recogniser, policy (no images in the loop)."""
    cfg, mcfg, tcfg = c.sim, c.model, c.train
    if ensemble is not None:
        mcfg = mcfg.__class__(**{**mcfg.__dict__, "ensemble": ensemble})
    t0 = time.time()
    k = jax.random.split(key, 5)
    data = estimators.generate(k[0], cfg, mcfg, n_missions or tcfg.n_train_missions, tcfg.T_train)
    norm = estimators.norm_from(data)
    params = eagent.init_params(k[1], cfg, mcfg, norm, n_hidden=n_hidden, cnn_J=cnn_J)
    params, losses = estimators.train_estimators(k[2], params, data, tcfg)
    log(f"  estimators trained: {losses}  ({time.time() - t0:.0f}s)")
    cnn, ls = recogniser.train_cnn(k[3], params.cnn, data, cfg, tcfg)
    params = params._replace(cnn=cnn)
    acc = float(recogniser.accuracy(jax.random.fold_in(k[3], 9), cnn, data, cfg))
    log(f"  recogniser trained: final loss {float(ls[-1]):.3f}, accuracy on training poses {acc:.3f}  ({time.time() - t0:.0f}s)")
    A = eagent.bind(cfg, mcfg, use_images=False, use_post=False)
    params, info = policy.train_policy(k[4], params, A, cfg, tcfg, lam=0.0, with_images=False, steps=pol_steps)
    log(f"  policy trained: best loss {info['best_loss']:.3f} (restart {info['best_restart']})  ({time.time() - t0:.0f}s)")
    return params, mcfg, {"est_losses": losses, "cnn_loss": float(ls[-1]), "cnn_acc": acc, "pol": info["best_loss"],
                          "train_seconds": time.time() - t0, "data_rows": int(data.cls.shape[0])}


def evaluate(agent, params, c, missions, method, seed, chunk=250, lam=0.0):
    """Rollout on the evaluation missions; returns (rows, logs) where rows are per-mission dicts including the
    Procedure EQF(t) results for the whole threshold grid, with eps0 calibrated on this method's own proxies."""
    cfg, ex = c.sim, c.exp
    T = cfg.T
    logs, term, total = sim.rollout_chunked(agent, params, missions, cfg, T, lam=lam, with_images=True, chunk=chunk)
    t_cal = min(100, T - 1)
    eps0 = procedure.calibrate_eps(logs.proxies, t_cal, 0.5)
    grid = procedure.stop_grid(logs.proxies, eps0, ex.eps_scales, ex.T_cons, ex.t_hat)
    stop_def, stopped_def = grid[(1.0, int(ex.T_cons[len(ex.T_cons) // 2]))]
    M = missions.x0.shape[0]
    rows = []
    at = lambda v: np.asarray(procedure.at_stop(v, stop_def))
    cost_cum = jnp.cumsum(logs.cost, axis=1)
    for m in range(M):
        r = {"method": method, "seed": seed, "mission": m, "cls": int(missions.cls[m]),
             "om_norm": float(jnp.linalg.norm(missions.f[m, :3])), "cd": float(missions.f[m, 3]),
             "b": float(missions.g[m, 0]), "s": float(missions.g[m, 1]),
             "total_cost": float(total[m]), "terminal": float(term[m]), "cost_per_stage": float(jnp.mean(logs.cost[m])),
             "final_dist": float(logs.dist[m, -1]), "min_dist": float(jnp.min(logs.dist[m])),
             "err_xp_T": float(logs.err_xp[m, -1]), "err_xa_T": float(logs.err_xa[m, -1]), "err_xv_T": float(logs.err_xv[m, -1]),
             "err_zp_T": float(logs.err_zp[m, -1]), "err_zv_T": float(logs.err_zv[m, -1]),
             "err_f_T": float(logs.err_f[m, -1]), "err_g_T": float(logs.err_g[m, -1]),
             "class_ok_T": float(logs.class_ok[m, -1]), "post_true_T": float(logs.post_true[m, -1]),
             "t_first_correct": int(jnp.argmax(logs.class_ok[m] > 0)) if bool(jnp.any(logs.class_ok[m] > 0)) else -1,
             "stop_t": int(stop_def[m]), "stopped": int(stopped_def[m]),
             "dist_stop": float(at(logs.dist)[m]), "err_zp_stop": float(at(logs.err_zp)[m]), "err_f_stop": float(at(logs.err_f)[m]),
             "err_g_stop": float(at(logs.err_g)[m]), "class_ok_stop": float(at(logs.class_ok)[m]),
             "cost_to_stop": float(np.asarray(procedure.at_stop(cost_cum, stop_def))[m])}
        for i, e in enumerate(np.asarray(eps0)):
            r[f"eps0_{i}"] = float(e)
        for (s_, Tc), (st, sp) in grid.items():
            r[f"stop_s{s_:g}_T{Tc}"] = int(st[m])
        rows.append(r)
    return rows, logs


def save_stage_logs(out, name, logs, n):
    sel = jax.tree_util.tree_map(lambda a: np.asarray(a[:n]).astype(np.float16), logs)
    np.savez_compressed(os.path.join(out, f"stage_logs_{name}.npz"), **sel._asdict())


def save_trajectories(out, name, logs, n, method, seed):
    rows = []
    for m in range(min(n, logs.p_x.shape[0])):
        for t in range(logs.p_x.shape[1]):
            rows.append({"method": method, "seed": seed, "mission": m, "t": t,
                         "px": float(logs.p_x[m, t, 0]), "py": float(logs.p_x[m, t, 1]), "pz": float(logs.p_x[m, t, 2]),
                         "zx": float(logs.p_z[m, t, 0]), "zy": float(logs.p_z[m, t, 1]), "zz": float(logs.p_z[m, t, 2]),
                         "dist": float(logs.dist[m, t]), "post_true": float(logs.post_true[m, t]),
                         "err_zp": float(logs.err_zp[m, t]), "err_f": float(logs.err_f[m, t])})
    io.append_csv(os.path.join(out, "trajectories_sample.csv"), rows)
