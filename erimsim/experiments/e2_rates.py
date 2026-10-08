"""E2: numerical check of the rates in the error bound. One-hidden-layer single networks (as in the theorem) of
width n, trained on M missions, evaluated on held-out missions and on the training missions (generalisation gap);
the CNN recogniser of depth J trained on M missions and evaluated by range. A high-capacity reference gives J(Γ°).
Units: est_n<n>_M<M> (loops over e2_seeds), cnn_J<J>_M<M>, ref."""
import os
import time
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from .. import io, sim, dynamics3d as dyn, models
from ..erim import agent as eagent, estimators, recogniser, policy
from . import common

TEST_SEED = 54321


def units(c):
    u = [f"est_n{n}_M{M}" for n in c.exp.n_grid for M in c.exp.M_grid]
    u += [f"cnn_J{J}_M{M}" for J in c.exp.J_grid for M in c.exp.M_grid]
    return u + ["ref"]


def _cfgs(c):
    mcfg = replace(c.model, depth=1, ensemble=1)
    tcfg = replace(c.train, est_steps=c.exp.e2_est_steps, pol_steps=c.exp.e2_pol_steps, pol_restarts=1,
                   cnn_steps=c.exp.e2_cnn_steps)
    return mcfg, tcfg


def train_eval_est(c, n, M, seed, log):
    cfg = c.sim
    mcfg, tcfg = _cfgs(c)
    key = jax.random.PRNGKey(seed * 100003 + n * 101 + M)
    k = jax.random.split(key, 4)
    data = estimators.generate(k[0], cfg, mcfg, M, tcfg.T_train)
    norm = estimators.norm_from(data)
    params = eagent.init_params(k[1], cfg, mcfg, norm, n_hidden=n)
    params, losses = estimators.train_estimators(k[2], params, data, tcfg)
    A = eagent.bind(cfg, mcfg, use_images=False, use_post=False)
    params, info = policy.train_policy(k[3], params, A, cfg, tcfg, steps=tcfg.pol_steps, batch=min(tcfg.pol_batch, 128), log=log)
    test = dyn.sample_missions(jax.random.PRNGKey(TEST_SEED), c.exp.e2_M_test, cfg)
    train_ms = dyn.sample_missions(k[0], M, cfg)                      # the missions the data came from
    f = jax.jit(lambda ms: sim.rollout(A, params, ms, cfg, cfg.T, 0.0, with_images=False))
    lt, _, tot_t = f(test)
    ltr, _, tot_tr = f(train_ms)
    row = {"n": n, "M": M, "seed": seed, "n_params": models.n_params(params.est_x) + models.n_params(params.pol),
           "cost_test": float(jnp.mean(tot_t)), "cost_train": float(jnp.mean(tot_tr)), "gap": float(jnp.mean(tot_t) - jnp.mean(tot_tr)),
           "err_xp_T": float(jnp.mean(lt.err_xp[:, -1])), "err_zp_T": float(jnp.mean(lt.err_zp[:, -1])),
           "err_zv_T": float(jnp.mean(lt.err_zv[:, -1])), "err_f_T": float(jnp.mean(lt.err_f[:, -1])),
           "err_g_T": float(jnp.mean(lt.err_g[:, -1])), "final_dist": float(jnp.mean(lt.dist[:, -1])),
           "pol_loss": info["best_loss"], "pol_initial": info["initial_loss"], "pol_improved": int(info["improved"]),
           "pol_skipped": info["n_skipped"], **{f"loss_{k_}": v for k_, v in losses.items()}}
    return row


def train_eval_cnn(c, J, M, seed, log):
    cfg = c.sim
    mcfg, tcfg = _cfgs(c)
    key = jax.random.PRNGKey(seed * 7919 + J * 31 + M)
    k = jax.random.split(key, 3)
    data = estimators.generate(k[0], cfg, mcfg, M, tcfg.T_train)
    cnn0 = models.init_cnn(k[1], J, mcfg.cnn_ch, mcfg.cnn_fc, 8)
    cnn, ls = recogniser.train_cnn(k[2], cnn0, data, cfg, tcfg)
    test = estimators.generate(jax.random.PRNGKey(TEST_SEED + 1), cfg, mcfg, c.exp.e2_M_test, tcfg.T_train)
    idx = jax.random.randint(jax.random.PRNGKey(5), (min(2048, test.cls.shape[0]),), 0, test.cls.shape[0])
    imgs = recogniser.render_batch(jax.random.PRNGKey(6), test, idx, cfg)
    pred = jnp.argmax(models.cnn(cnn, imgs), -1)
    ok = (pred == test.cls[idx]).astype(jnp.float32)
    r = jnp.linalg.norm(test.rel[idx], axis=-1)
    bins = {"near": r < 4, "mid": (r >= 4) & (r < 8), "far": r >= 8}
    row = {"J": J, "M": M, "seed": seed, "p_J": models.n_params(cnn), "loss": float(ls[-1]), "acc": float(jnp.mean(ok))}
    for name, msk in bins.items():
        row["acc_" + name] = float(jnp.sum(ok * msk) / jnp.maximum(jnp.sum(msk), 1))
    return row


def run_unit(c, unit, out, log=print):
    if unit.startswith("est_"):
        n = int(unit.split("_")[1][1:]); M = int(unit.split("_")[2][1:])
        rows = []
        for s in c.exp.e2_seeds:
            t0 = time.time()
            rows.append(train_eval_est(c, n, M, s, log))
            log(f"  est n={n} M={M} seed={s}: cost_test {rows[-1]['cost_test']:.1f} gap {rows[-1]['gap']:.1f} ({time.time() - t0:.0f}s)")
        io.append_csv(os.path.join(out, "rates_est.csv"), rows)
        return {"n": n, "M": M, "cost_test": float(np.mean([r["cost_test"] for r in rows]))}
    if unit.startswith("cnn_"):
        J = int(unit.split("_")[1][1:]); M = int(unit.split("_")[2][1:])
        rows = [train_eval_cnn(c, J, M, s, log) for s in c.exp.e2_seeds]
        io.append_csv(os.path.join(out, "rates_cnn.csv"), rows)
        return {"J": J, "M": M, "acc": float(np.mean([r["acc"] for r in rows]))}
    if unit == "ref":
        row = train_eval_est(c, c.exp.n_ref, c.exp.M_ref, 0, log)
        row_c = train_eval_cnn(c, c.exp.J_ref, c.exp.M_ref, 0, log)
        io.append_csv(os.path.join(out, "rates_ref.csv"), [{**row, **{"cnn_" + k: v for k, v in row_c.items()}}])
        return {"cost_ref": row["cost_test"], "acc_ref": row_c["acc"]}
    raise ValueError(unit)
