import jax, jax.numpy as jnp, numpy as np
from erimsim import sensors, dynamics3d as dyn
from erimsim.config import SimCfg

def test_rae_roundtrip():
    d = jnp.array([3.0, -4.0, 2.0])
    np.testing.assert_allclose(sensors.rae_to_rel(sensors.rel_to_rae(d)), d, atol=1e-6)

def test_object_sensor_bias_scale():
    cfg = SimCfg()
    m = dyn.sample_mission(jax.random.PRNGKey(2), cfg)
    y = sensors.observe_object(m.x0, m.z0, jnp.array([0.3, 1.1]), jnp.zeros(3), cfg)
    r = jnp.linalg.norm(m.z0[:3])
    np.testing.assert_allclose(y[0], 1.1 * r + 0.3, atol=1e-6)

def test_camera_axes_orthonormal():
    fwd, right, up = sensors.camera_axes(jnp.array([10.0, 0.7, -0.3]))
    for a, b in [(fwd, right), (fwd, up), (right, up)]:
        assert abs(float(jnp.dot(a, b))) < 1e-6
    assert abs(float(jnp.linalg.norm(up)) - 1) < 1e-6
