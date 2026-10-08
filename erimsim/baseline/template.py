"""Template recogniser: the eight solids rendered noise-free at the EKF-estimated relative pose (Object attitude
unknown to the classical chain, so the identity attitude is used), compared by silhouette Hamming distance, which is
invariant to the unknown lighting; the per-stage likelihood exp(-d / tau) is accumulated by Bayes' rule."""
import jax
import jax.numpy as jnp
from .. import render, quat, sensors


def log_likelihoods(img, rel_hat, y_obj, cfg, tau=0.02):
    fwd, right, up = sensors.camera_axes(y_obj)
    sil = render.silhouette(img)
    light = jnp.array([0.0, 0.0, 1.0])

    def dist(c):
        t = render.silhouette(render.render_clean(c, rel_hat, quat.identity(), fwd, right, up, light, cfg))
        return jnp.mean(jnp.abs(t - sil))

    d = jax.vmap(dist)(jnp.arange(render.N_CLASSES))
    return -d / tau


def update_posterior(log_post, img, rel_hat, y_obj, cfg):
    lp = log_post + log_likelihoods(img, rel_hat, y_obj, cfg)
    return lp - jax.scipy.special.logsumexp(lp)
