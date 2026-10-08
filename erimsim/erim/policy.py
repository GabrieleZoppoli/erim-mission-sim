"""Direct policy search (C_t): the policy parameters are optimised by stochastic gradient of the sampled mission
cost, differentiated through the known dynamics, the (frozen) estimators and, for the dual-effect variant, the
renderer and the recogniser. Horizon curriculum, gradient clipping, restarts keeping the best."""
import jax
import jax.numpy as jnp
from .. import sim, dynamics3d as dyn, models


def train_policy(key, params, Agent, cfg, tcfg, lam=0.0, with_images=False, steps=None, batch=None, H=None,
                 restarts=None, log_every=100):
    steps = steps or tcfg.pol_steps
    batch = batch or tcfg.pol_batch
    H = H or tcfg.pol_H
    restarts = restarts or tcfg.pol_restarts
    H0 = min(tcfg.pol_curriculum_start, H)

    def loss(pol, missions, h):
        p = params._replace(pol=pol)
        _, _, total = sim.rollout(Agent, p, missions, cfg, h, lam, with_images)
        return jnp.mean(total) / h

    grad_fn = jax.jit(jax.value_and_grad(loss), static_argnums=2)
    horizons = sorted({int(x) for x in jnp.linspace(H0, H, 4)})
    best = (jnp.inf, params.pol, None)
    hist = []
    for r in range(restarts):
        kr = jax.random.fold_in(key, r)
        pol = params.pol if r == 0 else jax.tree_util.tree_map(lambda a: a + 0.05 * jax.random.normal(jax.random.fold_in(kr, 1), a.shape), params.pol)
        st = models.adam_init(pol)
        for i in range(steps):
            h = horizons[min(len(horizons) - 1, (i * len(horizons)) // max(steps, 1))]
            missions = dyn.sample_missions(jax.random.fold_in(kr, 1000 + i), batch, cfg)
            l, g = grad_fn(pol, missions, h)
            pol, st = models.adam_update(pol, g, st, models.cosine_lr(i, steps, tcfg.pol_lr), clip=1.0)
            if i % log_every == 0 or i == steps - 1:
                hist.append((r, i, h, float(l)))
        missions = dyn.sample_missions(jax.random.fold_in(key, 7), batch, cfg)
        l_final, _ = grad_fn(pol, missions, H)
        if float(l_final) < float(best[0]):
            best = (l_final, pol, r)
    return params._replace(pol=best[1]), {"best_restart": best[2], "best_loss": float(best[0]), "history": hist}
