"""Direct policy search (C_t): the policy parameters are optimised by stochastic gradient of the sampled mission
cost, differentiated through the known dynamics, the (frozen) estimators and, for the dual-effect variant, the
renderer and the recogniser. Horizon curriculum, gradient clipping, restarts keeping the best.

Robustness (8 Oct 2026, after the first full-size run on the GPUs): the backward pass through 300 stages of a
closed loop that is not yet stabilising can overflow, and one non-finite step used to discard a whole restart. Now
every step is checked: a step whose loss or gradient is not finite is skipped (parameters and optimiser state
unchanged) and halves the learning rate; the policy returned is the one with the lowest finite validation loss at
the full horizon on a fixed batch, checked every val_every steps, never the last iterate."""
import math
import jax
import jax.numpy as jnp
from .. import sim, dynamics3d as dyn, models


def _global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(a * a) for a in jax.tree_util.tree_leaves(tree)))


def train_policy(key, params, Agent, cfg, tcfg, lam=0.0, with_images=False, steps=None, batch=None, H=None,
                 restarts=None, log_every=100, val_every=None, log=None, max_skips=50):
    """Returns (params with the best policy, info). info["improved"] is False when no step produced a finite
    validation loss below that of the initial policy, in which case the initial policy is returned unchanged."""
    steps = steps or tcfg.pol_steps
    batch = batch or tcfg.pol_batch
    H = H or tcfg.pol_H
    restarts = restarts or tcfg.pol_restarts
    H0 = min(tcfg.pol_curriculum_start, H)
    val_every = val_every or max(50, steps // 20)
    say = log or (lambda s: None)

    def loss(pol, missions, h):
        p = params._replace(pol=pol)
        _, _, total = sim.rollout(Agent, p, missions, cfg, h, lam, with_images)
        return jnp.mean(total) / h

    def step_fn(pol, st, missions, h, lr):
        l, g = jax.value_and_grad(loss)(pol, missions, h)
        gn = _global_norm(g)
        ok = jnp.isfinite(l) & jnp.isfinite(gn)
        g = jax.tree_util.tree_map(lambda a: jnp.where(ok, a, 0.0), g)
        pol1, st1 = models.adam_update(pol, g, st, lr, clip=1.0)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)
        return keep(pol1, pol), keep(st1, st), l, gn, ok

    step = jax.jit(step_fn, static_argnums=3)
    val_fn = jax.jit(loss, static_argnums=2)
    missions_val = dyn.sample_missions(jax.random.fold_in(key, 7), batch, cfg)
    horizons = sorted({int(x) for x in jnp.linspace(H0, H, 4)})
    l_init = float(val_fn(params.pol, missions_val, H))
    best = (l_init if math.isfinite(l_init) else math.inf, params.pol, None, -1)
    hist, n_skipped = [], 0
    say(f"    policy search: {steps} steps x {restarts} restarts, batch {batch}, horizons {horizons}, "
        f"validation loss of the initial policy {l_init:.3f}")
    for r in range(restarts):
        kr = jax.random.fold_in(key, r)
        pol = params.pol if r == 0 else jax.tree_util.tree_map(
            lambda a: a + 0.05 * jax.random.normal(jax.random.fold_in(kr, 1), a.shape), params.pol)
        st = models.adam_init(pol)
        lr_scale, skips, finite_run = 1.0, 0, 0
        for i in range(steps):
            h = horizons[min(len(horizons) - 1, (i * len(horizons)) // max(steps, 1))]
            missions = dyn.sample_missions(jax.random.fold_in(kr, 1000 + i), batch, cfg)
            lr = models.cosine_lr(i, steps, tcfg.pol_lr) * lr_scale
            pol, st, l, gn, ok = step(pol, st, missions, h, lr)
            if bool(ok):
                finite_run += 1
                if finite_run >= 100 and lr_scale < 1.0:        # a rare singular step must not freeze the search
                    lr_scale, finite_run = min(1.0, lr_scale * 2.0), 0
            else:
                finite_run = 0
                skips += 1
                n_skipped += 1
                lr_scale = max(lr_scale * 0.5, 1.0 / 64)
                say(f"    restart {r} step {i} h={h}: non-finite loss/gradient (loss {float(l):.3g}, |g| {float(gn):.3g}); "
                    f"step skipped, learning rate x{lr_scale:g}")
                if skips >= max_skips:
                    say(f"    restart {r} abandoned after {skips} skipped steps")
                    break
            if (i + 1) % val_every == 0 or i == steps - 1:
                lv = float(val_fn(pol, missions_val, H))
                if math.isfinite(lv) and lv < best[0]:
                    best = (lv, pol, r, i)
                say(f"    restart {r} step {i} h={h}: loss {float(l):.3f} |g| {float(gn):.3g} "
                    f"validation@{H} {lv:.3f} (best {best[0]:.3f})")
                hist.append((r, i, h, float(l), lv))
            elif i % log_every == 0:
                hist.append((r, i, h, float(l), None))
    improved = best[2] is not None
    if not improved:
        say("    WARNING: no step improved on the initial policy; returning the initial policy")
    return params._replace(pol=best[1]), {"best_restart": best[2], "best_step": best[3], "best_loss": float(best[0]),
                                          "initial_loss": l_init, "improved": improved, "n_skipped": n_skipped,
                                          "history": hist}
