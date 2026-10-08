import jax, jax.numpy as jnp, numpy as np
from erimsim import sim, dynamics3d as dyn, quat
from erimsim.config import Cfg, smoke
from erimsim.erim import agent as eagent, estimators, recogniser, policy


def test_from_two_columns_roundtrip():
    q = quat.random(jax.random.PRNGKey(4), (5,))
    R = quat.to_rotmat(q)
    q2 = quat.from_two_columns(R[..., :, 0], R[..., :, 1])
    np.testing.assert_allclose(quat.to_rotmat(q2), R, atol=1e-4)


def test_erim_pipeline_smoke():
    c = smoke(Cfg())
    cfg, mcfg, tcfg = c.sim, c.model, c.train
    key = jax.random.PRNGKey(0)
    data = estimators.generate(key, cfg, mcfg, tcfg.n_train_missions, tcfg.T_train)
    assert data.feats.shape == (tcfg.n_train_missions * tcfg.T_train, (mcfg.N + 1) * 24)
    assert bool(jnp.all(jnp.isfinite(data.feats))) and bool(jnp.all(jnp.isfinite(data.tx)))
    norm = estimators.norm_from(data)
    params = eagent.init_params(jax.random.PRNGKey(1), cfg, mcfg, norm)
    params, losses = estimators.train_estimators(jax.random.PRNGKey(2), params, data, tcfg)
    assert all(np.isfinite(v) for v in losses.values())
    cnn, ls = recogniser.train_cnn(jax.random.PRNGKey(3), params.cnn, data, cfg, tcfg)
    params = params._replace(cnn=cnn)
    assert bool(jnp.isfinite(ls[-1]))
    acc = recogniser.accuracy(jax.random.PRNGKey(4), cnn, data, cfg, n=64)
    assert 0.0 <= float(acc) <= 1.0
    A = eagent.bind(cfg, mcfg, use_images=False, use_post=False)
    params, info = policy.train_policy(jax.random.PRNGKey(5), params, A, cfg, tcfg, steps=3, batch=4, H=12, restarts=1)
    assert np.isfinite(info["best_loss"])
    A_full = eagent.bind(cfg, mcfg, use_images=True, use_post=False)
    ms = dyn.sample_missions(jax.random.PRNGKey(6), 3, cfg)
    logs, term, total = jax.jit(lambda m: sim.rollout(A_full, params, m, cfg, cfg.T))(ms)
    assert logs.cost.shape == (3, cfg.T) and bool(jnp.all(jnp.isfinite(total)))
    assert bool(jnp.all(logs.proxies >= 0))
