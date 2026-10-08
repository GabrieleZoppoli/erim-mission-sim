import jax, jax.numpy as jnp, numpy as np
from erimsim import procedure


def test_stop_times_match_sequential_reference():
    key = jax.random.PRNGKey(0)
    M, T = 50, 120
    # proxies that decay with time plus noise, so missions stop at different stages
    t = jnp.arange(T)[None, :, None]
    base = jnp.exp(-t / 40.0) * jax.random.uniform(key, (M, 1, 5), minval=0.5, maxval=2.0)
    proxies = base + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (M, T, 5))
    eps = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
    for Tc, th in [(4, 20), (8, 20), (16, 5)]:
        stop, stopped = procedure.stop_times(proxies, eps, Tc, th)
        ref = procedure.sequential_reference(proxies, eps, Tc, th)
        np.testing.assert_array_equal(np.asarray(stop), ref)
        np.testing.assert_array_equal(np.asarray(stopped), ref < T)


def test_grid_and_at_stop():
    proxies = jnp.ones((3, 30, 5)) * jnp.array([1.0, 0.1, 0.01])[:, None, None]
    grid = procedure.stop_grid(proxies, jnp.full(5, 0.5), (0.5, 1.0), (4,), 5)
    stop, stopped = grid[(1.0, 4)]
    # proxies below threshold from stage 0: the run reaches 4 at stage 3, the first admissible stage is t_hat = 5
    assert stop.tolist() == [30, 5, 5] and stopped.tolist() == [False, True, True]
    vals = jnp.arange(30)[None, :].repeat(3, 0).astype(jnp.float32)
    np.testing.assert_allclose(procedure.at_stop(vals, stop), [29, 5, 5])


def test_calibrate_eps_never_degenerates_to_zero():
    """A proxy that is exactly zero at the calibration stage (a saturated posterior) must still allow stopping."""
    M, T = 20, 60
    P = np.full((M, T, 5), 0.1, np.float32)
    P[:, :, 4] = 0.0                                   # recogniser always certain
    P[:, 40:, :4] = 0.01                               # estimators converge at stage 40
    eps = procedure.calibrate_eps(jnp.asarray(P), 50, 0.75)
    assert abs(float(eps[4]) - procedure.RECOGNITION_EPS) < 1e-6 and bool(jnp.all(eps > 0))
    stop, stopped = procedure.stop_times(jnp.asarray(P), eps, 4, 5)
    assert bool(jnp.all(stopped)) and int(stop[0]) == 43
