"""Observation channels. Own-state observation y_own (13): noisy p, attitude, v, w. Object sensor y_obj (3):
range * s + b, azimuth, elevation in the world frame, noise sd growing with range. The camera points along the
*observed* line of sight (gimbal driven by the sensor), so the pointing error moves the Object in the image."""
import jax.numpy as jnp
from . import quat


def observe_own(x, eta, cfg):
    """eta: (12,) standard normal -> y_own (13,)"""
    p, q, v, w = x[..., 0:3], x[..., 3:7], x[..., 7:10], x[..., 10:13]
    yp = p + cfg.sd_yp * eta[..., 0:3]
    yq = quat.boxplus(q, cfg.sd_yq * eta[..., 3:6])
    yv = v + cfg.sd_yv * eta[..., 6:9]
    yw = w + cfg.sd_yw * eta[..., 9:12]
    return jnp.concatenate([yp, yq, yv, yw], -1)


def rel_to_rae(d):
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-12)
    az = jnp.arctan2(d[..., 1], d[..., 0])
    # elevation through atan2 of (z, horizontal distance): arcsin(z / r) has an infinite derivative when the Object
    # is exactly above or below the Machine, which poisoned about one policy-search step in 200 on the GPUs
    el = jnp.arctan2(d[..., 2], jnp.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + 1e-12))
    return jnp.stack([r, az, el], -1)


def rae_to_rel(y):
    r, az, el = y[..., 0], y[..., 1], y[..., 2]
    return jnp.stack([r * jnp.cos(el) * jnp.cos(az), r * jnp.cos(el) * jnp.sin(az), r * jnp.sin(el)], -1)


def observe_object(x, z, g, eta, cfg):
    """eta: (3,) standard normal -> y_obj (3,) = (s r + b + n_r, az + n_a, el + n_e)"""
    d = z[..., 0:3] - x[..., 0:3]
    rae = rel_to_rae(d)
    r = rae[..., 0]
    grow = 1.0 + r / cfg.r0
    sd = jnp.stack([cfg.sd_r0 * grow, cfg.sd_a0 * grow, cfg.sd_a0 * grow], -1)
    y = rae + sd * eta
    y = y.at[..., 0].set(g[..., 1] * rae[..., 0] + g[..., 0] + sd[..., 0] * eta[..., 0])
    return y


def camera_axes(y_obj):
    """Camera frame from the observed azimuth/elevation: forward along the line of sight, right = forward x up_world."""
    az, el = y_obj[..., 1], y_obj[..., 2]
    fwd = jnp.stack([jnp.cos(el) * jnp.cos(az), jnp.cos(el) * jnp.sin(az), jnp.sin(el)], -1)
    up_w = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), fwd.shape)
    right = jnp.cross(fwd, up_w)
    rn = jnp.linalg.norm(right, axis=-1, keepdims=True)
    right = jnp.where(rn > 1e-6, right / jnp.maximum(rn, 1e-12), jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), fwd.shape))
    up = jnp.cross(right, fwd)
    return fwd, right, up
