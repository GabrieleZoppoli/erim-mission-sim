import jax, jax.numpy as jnp, numpy as np
from erimsim import quat, dynamics3d as dyn
from erimsim.config import SimCfg

def test_free_object_conserves_speed_and_turns_about_om():
    cfg = SimCfg()
    key = jax.random.PRNGKey(1)
    m = dyn.sample_mission(key, cfg)
    f = m.f.at[3].set(0.0)           # no drag
    z = m.z0
    om = f[:3]
    for t in range(50):
        z = dyn.object_step(z, f, jnp.zeros(6), cfg)
    v0, v = m.z0[7:10], z[7:10]
    np.testing.assert_allclose(jnp.linalg.norm(v), jnp.linalg.norm(v0), rtol=1e-5)
    # component of v along om is invariant under rotation about om
    np.testing.assert_allclose(jnp.dot(v, om), jnp.dot(v0, om), atol=1e-5)
    # the world angular velocity stays constant and equals R(q) om_body by construction
    assert float(jnp.linalg.norm(z[10:13] - om)) < 1e-6

def test_machine_accelerates_along_body_axis():
    cfg = SimCfg()
    x = dyn.join_x(jnp.zeros(3), quat.exp_map(jnp.array([0.0, 0.0, jnp.pi / 2])), jnp.zeros(3), jnp.zeros(3))
    u = jnp.array([1.0, 0, 0, 0, 0, 0])
    for _ in range(10):
        x = dyn.machine_step(x, u, jnp.zeros(6), cfg)
    v = x[7:10]
    assert v[1] > 0.9 and abs(v[0]) < 1e-5     # body +x is world +y after a 90 degree yaw
    assert abs(float(jnp.linalg.norm(x[3:7])) - 1) < 1e-6

def test_clip_and_mission_shapes():
    cfg = SimCfg()
    ms = dyn.sample_missions(jax.random.PRNGKey(0), 7, cfg)
    assert ms.x0.shape == (7, 13) and ms.z0.shape == (7, 13) and ms.f.shape == (7, 4) and ms.g.shape == (7, 2)
    r = jnp.linalg.norm(ms.z0[:, :3], axis=1)
    assert float(r.min()) >= cfg.r_min - 1e-6 and float(r.max()) <= cfg.r_max + 1e-6
    u = dyn.clip_u(jnp.full(6, 9.0), cfg)
    np.testing.assert_allclose(u, [2, 2, 2, 1, 1, 1])
