"""The best-iterate bookkeeping of ppo.train: it must return a valid second parameter set chosen inside the run and
must not alter the training trajectory (same seed, same final parameters with and without the bookkeeping being
consulted, since it lives outside the jitted update)."""
import jax
import jax.numpy as jnp
import numpy as np
from erimsim import config
from erimsim.rl import ppo


def _cfg():
    c = config.smoke(config.Cfg())
    return c


def test_train_returns_final_and_best_iterate():
    c = _cfg()
    p, curve, best = ppo.train(jax.random.PRNGKey(1), c.sim, c.model, total_steps=1024, n_envs=8, unroll=8, log=lambda *_: None)
    n_updates = 1024 // 64
    assert best["n_updates"] == n_updates and 0 <= best["update"] < n_updates
    assert np.isfinite(best["reward_smooth"])
    for leaf in jax.tree_util.tree_leaves(best["params"]):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert jax.tree_util.tree_structure(best["params"]) == jax.tree_util.tree_structure(p)
    assert len(curve) >= 2 and curve[-1]["update"] == n_updates - 1


def test_training_is_deterministic_for_a_seed():
    c = _cfg()
    p1, _, b1 = ppo.train(jax.random.PRNGKey(3), c.sim, c.model, total_steps=512, n_envs=8, unroll=8, log=lambda *_: None)
    p2, _, b2 = ppo.train(jax.random.PRNGKey(3), c.sim, c.model, total_steps=512, n_envs=8, unroll=8, log=lambda *_: None)
    for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert b1["update"] == b2["update"]
