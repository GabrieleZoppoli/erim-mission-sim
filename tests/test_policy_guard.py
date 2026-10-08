"""The policy search must survive non-finite steps: the optimiser never ingests inf/NaN, skipped steps leave the
parameters unchanged, and the returned policy is the best finite validation iterate, never a poisoned one."""
import math
import jax
import jax.numpy as jnp
from erimsim import models, sim, dynamics3d as dyn
from erimsim.config import Cfg, smoke
from erimsim.erim import agent as eagent, estimators, policy


def test_adam_update_ignores_non_finite_gradients():
    p = {"w": jnp.ones((3,))}
    st = models.adam_init(p)
    g = {"w": jnp.array([jnp.inf, jnp.nan, 1.0])}
    p1, st1 = models.adam_update(p, g, st, 1e-2, clip=1.0)
    assert bool(jnp.all(jnp.isfinite(p1["w"]))) and bool(jnp.all(jnp.isfinite(st1["v"]["w"])))
    assert bool(jnp.all(jnp.isfinite(st1["m"]["w"])))


def test_policy_search_survives_non_finite_steps(monkeypatch):
    c = smoke(Cfg())
    cfg, mcfg, tcfg = c.sim, c.model, c.train
    data = estimators.generate(jax.random.PRNGKey(2), cfg, mcfg, 4, 10)
    params = eagent.init_params(jax.random.PRNGKey(3), cfg, mcfg, estimators.norm_from(data))
    A = eagent.bind(cfg, mcfg, use_images=False, use_post=False)
    batch, H = 4, 10
    # a key whose validation batch (fold_in(key, 7)) is not poisoned by the rule below
    key = next(k for k in (jax.random.PRNGKey(s) for s in range(40))
               if float(dyn.sample_missions(jax.random.fold_in(k, 7), batch, cfg).z0[0, 0]) <= 0.0)
    real = sim.rollout

    def poisoned(agent, p, missions, cfg_, T, lam=0.0, with_images=True, extra=None):
        logs, term, total = real(agent, p, missions, cfg_, T, lam, with_images)
        bad = missions.z0[0, 0] > 0.0                      # about half of the training batches
        return logs, term, jnp.where(bad, jnp.nan, total)

    monkeypatch.setattr(policy.sim, "rollout", poisoned)
    out, info = policy.train_policy(key, params, A, cfg, tcfg, steps=16, batch=batch, H=H, restarts=1, val_every=4)
    assert info["n_skipped"] > 0, "the poisoned batches should have produced skipped steps"
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree_util.tree_leaves(out.pol))
    assert math.isfinite(info["best_loss"])
