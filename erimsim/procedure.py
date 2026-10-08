"""Procedure EQF(t): the mission stops at the first stage t >= t_hat at which all online quality proxies have been
below their thresholds for T_cons consecutive stages. Proxies are logged by the rollouts (they do not feed back into
the control), so the stopping rule is applied afterwards, vectorised over a whole grid of threshold settings.

Proxy order (5): x-estimate, z-estimate, f-identification, g-identification, recognition (1 - max posterior).
Baseline proxies are covariance traces; ERIM proxies are ensemble disagreements (trace of the ensemble covariance)."""
import jax
import jax.numpy as jnp
import numpy as np

N_PROXY = 5


def stop_times(proxies, eps, T_cons, t_hat):
    """proxies: [M, T, 5]; eps: [5]; returns stop_t [M] (int, T if never) and stopped [M] (bool)."""
    below = jnp.all(proxies < eps, axis=-1)                       # [M, T]

    def scan_fn(carry, b):
        run = jnp.where(b, carry + 1, 0)
        return run, run

    _, runs = jax.lax.scan(scan_fn, jnp.zeros(proxies.shape[0], jnp.int32), jnp.swapaxes(below, 0, 1))
    runs = jnp.swapaxes(runs, 0, 1)                                 # [M, T]
    T = proxies.shape[1]
    t_idx = jnp.arange(T)
    ok = (runs >= T_cons) & (t_idx[None, :] >= t_hat)
    stop = jnp.where(jnp.any(ok, axis=1), jnp.argmax(ok, axis=1), T)
    return stop, jnp.any(ok, axis=1)


def stop_grid(proxies, eps0, eps_scales, T_cons_list, t_hat):
    """Stop times for every (scale, T_cons): dict[(scale, T_cons)] -> (stop_t [M], stopped [M])."""
    out = {}
    for s in eps_scales:
        for Tc in T_cons_list:
            out[(float(s), int(Tc))] = stop_times(proxies, jnp.asarray(eps0) * float(s), int(Tc), int(t_hat))
    return out


def calibrate_eps(proxies, stage, quantile=0.5):
    """Reference thresholds: the given quantile of each proxy over missions at a fixed stage (baseline run)."""
    return jnp.quantile(proxies[:, stage, :], quantile, axis=0)


def sequential_reference(proxies, eps, T_cons, t_hat):
    """Plain Python re-implementation used by the tests."""
    P = np.asarray(proxies)
    out = []
    for m in range(P.shape[0]):
        run, stop = 0, P.shape[1]
        for t in range(P.shape[1]):
            run = run + 1 if np.all(P[m, t] < np.asarray(eps)) else 0
            if t >= t_hat and run >= T_cons:
                stop = t
                break
        out.append(stop)
    return np.array(out)


def at_stop(values, stop_t):
    """values [M, T, ...] -> value at the stop stage (or the last stage if never stopped)."""
    T = values.shape[1]
    idx = jnp.minimum(stop_t, T - 1)
    return jnp.take_along_axis(values, idx[:, None, *([None] * (values.ndim - 2))], axis=1)[:, 0]
