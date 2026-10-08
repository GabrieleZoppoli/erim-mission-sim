"""The mission rollout shared by every agent (classical, ERIM, RL), vectorised over missions with vmap and over
stages with lax.scan. Every random draw of a mission at stage t comes from fold_in(mission.key, t), so two agents
evaluated on the same missions see identical noise realisations (common random numbers). Rollouts never stop early:
Procedure EQF(t) is applied afterwards to the logged proxies (procedure.py), which is valid because stopping does not
feed back into the control.

An agent is any object with three pure functions:
    init(params, obs0, cfg) -> S
    update(params, S, obs, u_prev, t) -> (S, Summary)
    act(params, summary, obs, t, key) -> u (6,)
where obs = (y_own 13, y_obj 3, img [H, W]). The Summary carries the estimates the logs and the cost need."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
from . import quat, dynamics3d as dyn, sensors, render


class Summary(NamedTuple):
    x_hat: jnp.ndarray      # (13,) p, q, v, w
    z_hat: jnp.ndarray      # (6,)  p_z, v_z
    f_hat: jnp.ndarray      # (4,)
    g_hat: jnp.ndarray      # (2,)
    log_post: jnp.ndarray   # (K,) normalised class log-posterior
    proxies: jnp.ndarray    # (5,) online quality proxies


class StageLog(NamedTuple):
    cost: jnp.ndarray
    dist: jnp.ndarray
    err_xp: jnp.ndarray
    err_xa: jnp.ndarray
    err_xv: jnp.ndarray
    err_zp: jnp.ndarray
    err_zv: jnp.ndarray
    err_f: jnp.ndarray
    err_g: jnp.ndarray
    class_ok: jnp.ndarray
    post_true: jnp.ndarray
    entropy: jnp.ndarray
    proxies: jnp.ndarray    # (5,)
    u_norm: jnp.ndarray
    y_r: jnp.ndarray        # observed range (used by the common-random-numbers test)
    p_x: jnp.ndarray        # (3,) true Machine position (sample trajectories)
    p_z: jnp.ndarray        # (3,) true Object position


def entropy(log_post):
    p = jnp.exp(log_post)
    return -jnp.sum(p * log_post)


def observe(mission, x, z, t, key, cfg, with_images=True):
    k1, k2, k3 = jax.random.split(key, 3)
    y_own = sensors.observe_own(x, jax.random.normal(k1, (12,)), cfg)
    y_obj = sensors.observe_object(x, z, mission.g, jax.random.normal(k2, (3,)), cfg)
    if with_images:
        fwd, right, up = sensors.camera_axes(y_obj)
        img = render.render(mission.cls, z[0:3] - x[0:3], z[3:7], fwd, right, up, mission.light, k3, cfg)
    else:
        img = jnp.zeros((cfg.img, cfg.img))
    return y_own, y_obj, img


def stage_cost(x, z, u, log_post, cfg, lam):
    dist2 = jnp.sum((z[0:3] - x[0:3]) ** 2)
    return dist2 + cfg.c_u * jnp.sum(u * u) + lam * entropy(log_post)


def rollout_one(agent, params, mission, cfg, T, lam=0.0, with_images=True, extra=None):
    """Returns (StageLog over T stages, terminal cost, total cost[, extras]) where extras are the per-stage outputs
    of extra(S, sm, obs, x, z, mission, t) when given (used to collect training data)."""
    k0 = dyn.stage_key(mission.key, 0)
    obs0 = observe(mission, mission.x0, mission.z0, 0, jax.random.split(k0, 4)[0], cfg, with_images)
    S0 = agent.init(params, obs0, cfg)

    def step(carry, t):
        x, z, S, u_prev = carry
        key = dyn.stage_key(mission.key, t)
        kobs, kx, kz, kact = jax.random.split(key, 4)
        obs = observe(mission, x, z, t, kobs, cfg, with_images)
        S, sm = agent.update(params, S, obs, u_prev, t)
        u = dyn.clip_u(agent.act(params, sm, obs, t, kact), cfg)
        ex = extra(S, sm, obs, x, z, mission, t) if extra is not None else None
        c = stage_cost(x, z, u, sm.log_post, cfg, lam)
        p_hat, q_hat = sm.x_hat[0:3], quat.normalize(sm.x_hat[3:7])
        log = StageLog(
            cost=c, dist=jnp.linalg.norm(z[0:3] - x[0:3]),
            err_xp=jnp.linalg.norm(p_hat - x[0:3]), err_xa=jnp.linalg.norm(quat.boxminus(q_hat, x[3:7])),
            err_xv=jnp.linalg.norm(sm.x_hat[7:10] - x[7:10]),
            err_zp=jnp.linalg.norm(sm.z_hat[0:3] - z[0:3]), err_zv=jnp.linalg.norm(sm.z_hat[3:6] - z[7:10]),
            err_f=jnp.linalg.norm(sm.f_hat - mission.f), err_g=jnp.linalg.norm(sm.g_hat - mission.g),
            class_ok=(jnp.argmax(sm.log_post) == mission.cls).astype(jnp.float32),
            post_true=jnp.exp(sm.log_post[mission.cls]), entropy=entropy(sm.log_post),
            proxies=sm.proxies, u_norm=jnp.linalg.norm(u), y_r=obs[1][0], p_x=x[0:3], p_z=z[0:3])
        x1 = dyn.machine_step(x, u, jax.random.normal(kx, (6,)), cfg)
        z1 = dyn.object_step(z, mission.f, jax.random.normal(kz, (6,)), cfg)
        return (x1, z1, S, u), (log, ex)

    (xT, zT, _, _), (logs, extras) = lax.scan(step, (mission.x0, mission.z0, S0, jnp.zeros(6)), jnp.arange(T))
    terminal = cfg.c_term * jnp.sum((zT[0:3] - xT[0:3]) ** 2)
    total = jnp.sum(logs.cost) + terminal
    return (logs, terminal, total) if extra is None else (logs, terminal, total, extras)


def rollout(agent, params, missions, cfg, T, lam=0.0, with_images=True, extra=None):
    """Vectorised over missions: logs [M, T, ...], terminal [M], total [M]."""
    return jax.vmap(lambda m: rollout_one(agent, params, m, cfg, T, lam, with_images, extra))(missions)


def rollout_chunked(agent, params, missions, cfg, T, lam=0.0, with_images=True, chunk=250):
    """Same, in chunks of missions to bound memory on evaluation runs."""
    M = missions.x0.shape[0]
    f = jax.jit(lambda ms: rollout(agent, params, ms, cfg, T, lam, with_images))
    outs = []
    for i in range(0, M, chunk):
        sl = jax.tree_util.tree_map(lambda a: a[i:i + chunk], missions)
        outs.append(f(sl))
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, 0), *outs)


class TruthAgent:
    """Reference agent for tests: knows the true state, LQ-like proportional controller. Not used in experiments."""

    @staticmethod
    def init(params, obs0, cfg):
        return jnp.zeros(1)

    @staticmethod
    def update(params, S, obs, u_prev, t):
        y_own, y_obj, img = obs
        rel = sensors.rae_to_rel(y_obj)
        K = img.shape[0] if False else 8
        sm = Summary(x_hat=y_own, z_hat=jnp.concatenate([y_own[0:3] + rel, jnp.zeros(3)]), f_hat=jnp.zeros(4),
                     g_hat=jnp.array([0.0, 1.0]), log_post=jnp.full(K, -jnp.log(K)), proxies=jnp.ones(5))
        return S, sm

    @staticmethod
    def act(params, sm, obs, t, key):
        rel = sm.z_hat[0:3] - sm.x_hat[0:3]
        a_world = 0.5 * rel - 1.0 * sm.x_hat[7:10]
        a_body = quat.rotate(quat.conj(sm.x_hat[3:7]), a_world)
        return jnp.concatenate([a_body, -0.5 * sm.x_hat[10:13]])
