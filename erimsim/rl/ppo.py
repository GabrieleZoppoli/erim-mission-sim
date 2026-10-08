"""PPO with a CNN encoder on the image and an MLP on a normalised proprioceptive vector (relative Object position
and range from the sensor, own attitude, velocities, t/T). Vectorised environments built from the same dynamics,
sensors and renderer as the rest of the package; the reward is minus the stage cost scaled by 1/100 and, for the
advantages and the value targets, by a running estimate of the return scale (the usual reward normalisation);
episodes last T stages and reset to fresh missions. Evaluation of the trained actor goes through sim.rollout
(RLAgent), on the same missions as the other methods.

8 Oct 2026, after the first GPU run (final distance 526 m, reward spikes to -4500): the inputs were raw (absolute
positions of tens of metres), the value loss on raw returns dominated the shared features, the pre-tanh mean could
saturate the controls, and the evaluation fed u_prev = 0 where training fed the previous action. Now: normalised
inputs without u_prev, return scaling, a small penalty on the pre-tanh mean, clipped log-std."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
from .. import dynamics3d as dyn, sensors, sim, models, render
from ..sim import Summary

PROPRIO = 3 + 1 + 4 + 3 + 3 + 1
LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
MEAN_PENALTY = 1e-3


class PPOParams(NamedTuple):
    cnn: dict
    prop: list
    trunk: list
    mean: list
    log_std: jnp.ndarray
    value: list


def init_params(key, cfg, mcfg):
    k = jax.random.split(key, 6)
    return PPOParams(cnn=models.init_cnn(k[0], mcfg.cnn_J, mcfg.cnn_ch, mcfg.cnn_fc, 64),
                     prop=models.init_mlp(k[1], [PROPRIO, 64]), trunk=models.init_mlp(k[2], [128, 128, 128]),
                     mean=models.init_mlp(k[3], [128, 6], scale=0.1), log_std=jnp.full(6, -0.5),
                     value=models.init_mlp(k[4], [128, 1]))


def features(p, img, prop):
    f_img = jax.nn.relu(models.cnn(p.cnn, img))
    f_prop = jax.nn.relu(models.mlp(p.prop, prop))
    return jax.nn.relu(models.mlp(p.trunk, jnp.concatenate([f_img, f_prop], -1)))


def actor_critic(p, img, prop):
    h = features(p, img, prop)
    return models.mlp(p.mean, h), jnp.clip(p.log_std, LOG_STD_MIN, LOG_STD_MAX), models.mlp(p.value, h)[..., 0]


def proprio(y_own, y_obj, t, cfg):
    """Normalised proprioceptive input: Object position relative to the Machine (from the sensor) and range in
    units of 10 m, own attitude, own velocities in units of 5 m/s and 2 rad/s, normalised time."""
    rel = sensors.rae_to_rel(y_obj)
    return jnp.concatenate([rel / 10.0, y_obj[0:1] / 10.0, y_own[3:7], y_own[7:10] / 5.0, y_own[10:13] / 2.0,
                            jnp.array([t / cfg.T])])


def to_control(a, cfg):
    return jnp.concatenate([cfg.u_max * jnp.tanh(a[..., 0:3]), cfg.w_max * jnp.tanh(a[..., 3:6])], -1)


class EnvState(NamedTuple):
    mission: dyn.Mission
    x: jnp.ndarray
    z: jnp.ndarray
    t: jnp.ndarray
    u_prev: jnp.ndarray


def env_reset(key, cfg):
    m = dyn.sample_mission(key, cfg)
    return EnvState(m, m.x0, m.z0, jnp.zeros((), jnp.int32), jnp.zeros(6))


def env_obs(s, cfg):
    key = jax.random.split(dyn.stage_key(s.mission.key, s.t), 4)[0]
    y_own, y_obj, img = sim.observe(s.mission, s.x, s.z, s.t, key, cfg, True)
    return img, proprio(y_own, y_obj, s.t, cfg)


def env_step(s, a, key, cfg):
    u = to_control(a, cfg)
    key = dyn.stage_key(s.mission.key, s.t)
    _, kx, kz, _ = jax.random.split(key, 4)
    r = -sim.stage_cost(s.x, s.z, u, jnp.zeros(1), cfg, 0.0) / 100.0
    x1 = dyn.machine_step(s.x, u, jax.random.normal(kx, (6,)), cfg)
    z1 = dyn.object_step(s.z, s.mission.f, jax.random.normal(kz, (6,)), cfg)
    t1 = s.t + 1
    done = t1 >= cfg.T
    r = r - done * cfg.c_term * jnp.sum((z1[0:3] - x1[0:3]) ** 2) / 100.0
    s1 = EnvState(s.mission, x1, z1, t1, u)
    s_reset = env_reset(jax.random.fold_in(s.mission.key, 777), cfg)
    s1 = jax.tree_util.tree_map(lambda a_, b_: jnp.where(done, b_, a_), s1, s_reset)
    return s1, r, done


def gae(rewards, values, dones, last_value, gamma, lam):
    def body(carry, inp):
        r, v, d, v_next = inp
        delta = r + gamma * v_next * (1 - d) - v
        adv = delta + gamma * lam * (1 - d) * carry
        return adv, adv
    v_next = jnp.concatenate([values[1:], last_value[None]], 0)
    _, advs = lax.scan(body, jnp.zeros_like(last_value), (rewards, values, dones, v_next), reverse=True)
    return advs, advs + values


def train(key, cfg, mcfg, total_steps, n_envs, unroll, lr=3e-4, epochs=4, minibatches=8, clip=0.2, gamma=0.99,
          lam=0.95, ent=0.0, vcoef=0.5, log_every=10, log=print):
    k0, k1 = jax.random.split(key)
    p = init_params(k0, cfg, mcfg)
    st = models.adam_init(p)
    envs = jax.vmap(lambda k: env_reset(k, cfg))(jax.random.split(k1, n_envs))
    obs_fn = jax.vmap(lambda s: env_obs(s, cfg))
    n_updates = max(1, total_steps // (n_envs * unroll))

    def policy_step(p, s, key):
        img, prop = obs_fn(s)
        mean, log_std, v = jax.vmap(lambda i, q: actor_critic(p, i, q))(img, prop)
        a = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        logp = -0.5 * jnp.sum(((a - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), -1)
        return a, logp, v

    def collect(p, envs, key):
        def body(carry, k):
            s = carry
            a, logp, v = policy_step(p, s, k)
            s1, r, d = jax.vmap(lambda s_, a_, k_: env_step(s_, a_, k_, cfg))(s, a, jax.random.split(k, n_envs))
            return s1, (s, a, logp, v, r, d)
        envs1, traj = lax.scan(body, envs, jax.random.split(key, unroll))
        return envs1, traj

    def loss_fn(p, s_batch, a, logp_old, adv, ret):
        img, prop = obs_fn(s_batch)
        mean, log_std, v = jax.vmap(lambda i, q: actor_critic(p, i, q))(img, prop)
        logp = -0.5 * jnp.sum(((a - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), -1)
        ratio = jnp.exp(logp - logp_old)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - clip, 1 + clip) * adv_n))
        vl = jnp.mean((v - ret) ** 2)                       # returns are in scaled units (see update)
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        reg = MEAN_PENALTY * jnp.mean(mean ** 2)             # keeps the pre-tanh mean away from saturation
        return pg + vcoef * vl - ent * entropy + reg, (pg, vl)

    @jax.jit
    def update(p, st, envs, ret_var, key, i):
        kc, ku = jax.random.split(key)
        envs1, (s, a, logp, v, r, d) = collect(p, envs, kc)
        img_l, prop_l = obs_fn(envs1)
        _, _, last_v = jax.vmap(lambda i_, q: actor_critic(p, i_, q))(img_l, prop_l)
        scale = jnp.sqrt(ret_var + 1e-8)
        adv, ret = gae(r / scale, v, d, last_v, gamma, lam)   # value network and advantages in scaled units
        ret_var = jnp.where(i == 0, jnp.mean((ret * scale) ** 2), 0.99 * ret_var + 0.01 * jnp.mean((ret * scale) ** 2))
        flat = lambda x: x.reshape((unroll * n_envs,) + x.shape[2:])
        s_f = jax.tree_util.tree_map(flat, s)
        a_f, logp_f, adv_f, ret_f = flat(a), flat(logp), flat(adv), flat(ret)
        n = unroll * n_envs
        mb = n // minibatches

        def epoch(carry, ke):
            p, st = carry
            perm = jax.random.permutation(ke, n)

            def mini(carry, j):
                p, st = carry
                idx = lax.dynamic_slice(perm, (j * mb,), (mb,))
                take = lambda x: jnp.take(x, idx, axis=0)
                (l, (pg, vl)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, jax.tree_util.tree_map(take, s_f), take(a_f), take(logp_f), take(adv_f), take(ret_f))
                p, st = models.adam_update(p, g, st, lr * (1 - i / n_updates), clip=0.5)
                return (p, st), (pg, vl)
            (p, st), aux = lax.scan(mini, (p, st), jnp.arange(minibatches))
            return (p, st), aux
        (p, st), aux = lax.scan(epoch, (p, st), jax.random.split(ku, epochs))
        return p, st, envs1, ret_var, jnp.mean(r), jnp.mean(d), aux

    curve = []
    ret_var = jnp.ones(())
    for i in range(n_updates):
        p, st, envs, ret_var, r_mean, d_mean, aux = update(p, st, envs, ret_var, jax.random.fold_in(key, 10 + i), i)
        if i % log_every == 0 or i == n_updates - 1:
            curve.append({"update": i, "env_steps": (i + 1) * n_envs * unroll, "reward_mean": float(r_mean),
                          "pg_loss": float(jnp.mean(aux[0])), "v_loss": float(jnp.mean(aux[1])),
                          "return_scale": float(jnp.sqrt(ret_var))})
            log(f"  ppo update {i}/{n_updates}: mean reward {float(r_mean):.4f}  return scale {float(jnp.sqrt(ret_var)):.3g}")
    return p, curve


class RLAgent:
    """Wraps the trained actor for sim.rollout; it has no estimates, so the Summary carries raw observations."""
    cfg = None

    @classmethod
    def init(cls, params, obs0, cfg):
        return jnp.zeros(6)                      # u_prev

    @classmethod
    def update(cls, params, S, obs, u_prev, t):
        y_own, y_obj, img = obs
        rel = sensors.rae_to_rel(y_obj)
        sm = Summary(x_hat=y_own, z_hat=jnp.concatenate([y_own[0:3] + rel, jnp.zeros(3)]), f_hat=jnp.zeros(4),
                     g_hat=jnp.array([0.0, 1.0]), log_post=jnp.full(render.N_CLASSES, -jnp.log(render.N_CLASSES)),
                     proxies=jnp.ones(5))
        return u_prev, sm

    @classmethod
    def act(cls, params, sm, obs, t, key):
        y_own, y_obj, img = obs
        mean, _, _ = actor_critic(params, img, proprio(y_own, y_obj, t, cls.cfg))
        return to_control(mean, cls.cfg)


def bind(cfg):
    class A(RLAgent):
        pass
    A.cfg = cfg
    return A
