import jax, jax.numpy as jnp, numpy as np
from erimsim import sim, dynamics3d as dyn
from erimsim.config import SimCfg


def test_rollout_shapes_finite_and_truth_agent_approaches():
    cfg = SimCfg(T=60, img=16, march_steps=12)
    ms = dyn.sample_missions(jax.random.PRNGKey(0), 4, cfg)
    logs, term, total = jax.jit(lambda m: sim.rollout(sim.TruthAgent, None, m, cfg, cfg.T))(ms)
    assert logs.cost.shape == (4, cfg.T) and logs.proxies.shape == (4, cfg.T, 5) and total.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(total)))
    assert float(jnp.mean(logs.dist[:, -1])) < float(jnp.mean(logs.dist[:, 0])) * 0.5


def test_common_random_numbers_across_agents():
    cfg = SimCfg(T=20, img=16, march_steps=8)
    ms = dyn.sample_missions(jax.random.PRNGKey(3), 2, cfg)

    class ZeroAgent(sim.TruthAgent):
        @staticmethod
        def act(params, sm, obs, t, key):
            return jnp.zeros(6)

    la, _, _ = sim.rollout(sim.TruthAgent, None, ms, cfg, cfg.T)
    lb, _, _ = sim.rollout(ZeroAgent, None, ms, cfg, cfg.T)
    # the first observation is identical (same key, same state); later the states differ but the noise seeds do not
    np.testing.assert_allclose(la.y_r[:, 0], lb.y_r[:, 0])
    assert not bool(jnp.allclose(la.dist[:, -1], lb.dist[:, -1]))
