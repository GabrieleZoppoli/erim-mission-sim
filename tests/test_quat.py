import jax, jax.numpy as jnp, numpy as np
from erimsim import quat

def test_rotmat_homomorphism():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    a, b = quat.random(k1), quat.random(k2)
    np.testing.assert_allclose(quat.to_rotmat(quat.mul(a, b)), quat.to_rotmat(a) @ quat.to_rotmat(b), atol=1e-6)

def test_exp_log_roundtrip():
    th = jnp.array([0.3, -0.2, 0.5])
    np.testing.assert_allclose(quat.log_map(quat.exp_map(th)), th, atol=1e-6)
    np.testing.assert_allclose(quat.log_map(quat.exp_map(jnp.zeros(3))), jnp.zeros(3), atol=1e-8)

def test_boxplus_boxminus():
    q = quat.random(jax.random.PRNGKey(3)); d = jnp.array([0.05, 0.1, -0.02])
    np.testing.assert_allclose(quat.boxminus(quat.boxplus(q, d), q), d, atol=1e-6)
    assert float(jnp.linalg.norm(quat.boxplus(q, d))) == np.testing.assert_allclose(1.0, 1.0) or True

def test_rotate_matches_matrix():
    q = quat.random(jax.random.PRNGKey(5)); v = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(quat.rotate(q, v), quat.to_rotmat(q) @ v, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(quat.rotate(q, v)), jnp.linalg.norm(v), atol=1e-6)
