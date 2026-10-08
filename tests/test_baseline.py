import jax, jax.numpy as jnp, numpy as np
from erimsim import dynamics3d as dyn, sensors, sim
from erimsim.baseline import ekf, agent as bagent
from erimsim.config import SimCfg


def test_object_ekf_consistency_and_identification():
    """Object EKF alone, Machine fixed at the origin and known: NEES of the position block within loose chi-square
    bounds, position error far below the initial one, angular velocity partly identified."""
    cfg = SimCfg()
    key = jax.random.PRNGKey(7)
    ms = dyn.sample_missions(key, 40, cfg)

    def run(m):
        z = m.z0
        x = m.x0
        oe = ekf.object_init(sensors.observe_object(x, z, m.g, jax.random.normal(jax.random.fold_in(m.key, 0), (3,)), cfg), x[0:3], cfg)

        def step(carry, t):
            z, oe = carry
            k = jax.random.fold_in(m.key, t + 1)
            k1, k2 = jax.random.split(k)
            z1 = dyn.object_step(z, m.f, jax.random.normal(k1, (6,)), cfg)
            oe1 = ekf.object_predict(oe, cfg)
            y = sensors.observe_object(x, z1, m.g, jax.random.normal(k2, (3,)), cfg)
            oe1, _, _ = ekf.object_update(oe1, y, x[0:3], cfg)
            return (z1, oe1), None

        (zT, oeT), _ = jax.lax.scan(step, (z, oe), jnp.arange(200))
        e = oeT.zeta[0:3] - zT[0:3]
        nees = e @ jnp.linalg.solve(oeT.P[0:3, 0:3], e)
        return nees, jnp.linalg.norm(e), jnp.linalg.norm(oeT.zeta[6:9] - m.f[0:3]), jnp.linalg.norm(m.f[0:3])

    nees, err_p, err_om, om = jax.jit(jax.vmap(run))(ms)
    assert bool(jnp.all(jnp.isfinite(nees)))
    assert 0.5 < float(jnp.mean(nees)) < 12.0, float(jnp.mean(nees))
    # a static observer cannot separate the range bias (up to 0.5 m) from the range, so position errors stay ~ b
    assert float(jnp.median(err_p)) < 1.2
    assert float(jnp.mean(err_om)) < 0.8 * float(jnp.mean(om))


def test_baseline_agent_rollout():
    cfg = SimCfg(T=120, img=16, march_steps=12)
    ms = dyn.sample_missions(jax.random.PRNGKey(11), 4, cfg)
    A = bagent.bind(cfg)
    params = bagent.make_params(cfg)
    logs, term, total = jax.jit(lambda m: sim.rollout(A, params, m, cfg, cfg.T))(ms)
    assert bool(jnp.all(jnp.isfinite(total)))
    assert float(jnp.mean(logs.dist[:, -1])) < 0.5 * float(jnp.mean(logs.dist[:, 0]))
    assert float(jnp.mean(logs.err_xp[:, -1])) < 0.2
    assert bool(jnp.all(logs.proxies[:, -1, :] >= 0))
    assert float(jnp.mean(logs.post_true[:, -1])) > 1.0 / 8
