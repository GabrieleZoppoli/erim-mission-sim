"""estimators.generate: the rows do not depend on the chunk size (same missions, same order)."""
import jax
import jax.numpy as jnp
import numpy as np
from erimsim.config import Cfg, smoke
from erimsim.erim import estimators


def test_generate_rows_independent_of_chunk_size():
    c = smoke(Cfg())
    a = estimators.generate(jax.random.PRNGKey(5), c.sim, c.model, 5, 12, chunk=5)
    b = estimators.generate(jax.random.PRNGKey(5), c.sim, c.model, 5, 12, chunk=2)
    for fa, fb in zip(a, b):
        assert fa.shape == fb.shape and fa.shape[0] == 5 * 12
        assert np.allclose(np.asarray(fa), np.asarray(fb), rtol=1e-5, atol=1e-5)
