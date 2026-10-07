"""Rigid-body Machine and drifting Object in 3D.

Machine state x = [p(3), q(4), v(3), w(3)] (world position, attitude body->world, world velocity, body angular velocity).
Control u = [a_body(3), alpha_body(3)], bounded per axis.
Object state z = [p(3), q(4), v(3), om(3)]: v is the world velocity, om the constant world-frame angular velocity:
    v' = (1 - c_d dt) R(exp(om dt)) v + noise,   p' = p + dt v',   q' = q ⊗ exp(R(q)^T om dt).
The unknown dynamics parameters are f = (om(3), c_d); the unknown sensor parameters g = (range bias b, range scale s).
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import quat


class Mission(NamedTuple):
    x0: jnp.ndarray      # (13,)
    z0: jnp.ndarray      # (13,)
    cls: jnp.ndarray     # () int
    f: jnp.ndarray       # (4,) om(3), c_d
    g: jnp.ndarray       # (2,) b, s
    light: jnp.ndarray   # (3,) unit light direction (world)
    key: jnp.ndarray     # PRNG key, folded with the stage index for every random draw of the mission


def split_x(x):
    return x[..., 0:3], x[..., 3:7], x[..., 7:10], x[..., 10:13]


def join_x(p, q, v, w):
    return jnp.concatenate([p, q, v, w], -1)


def clip_u(u, cfg):
    lim = jnp.array([cfg.u_max] * 3 + [cfg.w_max] * 3)
    return jnp.clip(u, -lim, lim)


def machine_step(x, u, xi, cfg):
    """Semi-implicit Euler. xi = (xi_v(3), xi_w(3)) standard normal."""
    p, q, v, w = split_x(x)
    u = clip_u(u, cfg)
    a_world = quat.rotate(q, u[..., 0:3])
    v1 = v + cfg.dt * a_world + cfg.sd_v * xi[..., 0:3]
    p1 = p + cfg.dt * v1
    w1 = w + cfg.dt * u[..., 3:6] + cfg.sd_w * xi[..., 3:6]
    q1 = quat.boxplus(q, cfg.dt * w1)
    return join_x(p1, q1, v1, w1)


def object_step(z, f, xi, cfg):
    p, q, v, om = split_x(z)
    om = f[..., 0:3]
    cd = f[..., 3:4]
    Rom = quat.to_rotmat(quat.exp_map(cfg.dt * om))
    v1 = (1.0 - cd * cfg.dt) * jnp.einsum('...ij,...j->...i', Rom, v) + cfg.sd_vz * xi[..., 0:3]
    p1 = p + cfg.dt * v1
    om_body = quat.rotate(quat.conj(q), om)
    q1 = quat.boxplus(q, cfg.dt * om_body)
    return join_x(p1, q1, v1, om)


def sample_mission(key, cfg) -> Mission:
    k = jax.random.split(key, 10)
    d = jax.random.normal(k[0], (3,)); d = d / jnp.linalg.norm(d)
    r = jax.random.uniform(k[1], (), minval=cfg.r_min, maxval=cfg.r_max)
    pz = r * d
    qz = quat.random(k[2])
    vz = cfg.sd_vz0 * jax.random.normal(k[3], (3,))
    om = jax.random.uniform(k[4], (3,), minval=-cfg.om_max, maxval=cfg.om_max)
    cd = jax.random.uniform(k[5], (), minval=0.0, maxval=cfg.cd_max)
    cls = jax.random.randint(k[6], (), 0, cfg.n_classes)
    b = jax.random.uniform(k[7], (), minval=-cfg.b_max, maxval=cfg.b_max)
    s = jax.random.uniform(k[8], (), minval=1 - cfg.s_dev, maxval=1 + cfg.s_dev)
    light = jax.random.normal(k[9], (3,)); light = light / jnp.linalg.norm(light)
    x0 = join_x(jnp.zeros(3), quat.identity(), jnp.zeros(3), jnp.zeros(3))
    z0 = join_x(pz, qz, vz, om)
    return Mission(x0, z0, cls, jnp.concatenate([om, cd[None]]), jnp.stack([b, s]), light, key)


def sample_missions(key, M, cfg) -> Mission:
    return jax.vmap(lambda k: sample_mission(k, cfg))(jax.random.split(key, M))


def stage_key(mission_key, t):
    return jax.random.fold_in(mission_key, t)
