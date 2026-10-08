"""Training data and training of the ERIM estimators (P_x, P_z, P_f, P_g).

Data: exploratory missions driven by the classical chain with additive control noise (two thirds) or by random
controls (one third), logged through sim.rollout with an `extra` function that records, at every stage, the window
features and the four targets built by windows.features / windows.targets, plus the true relative pose and light,
from which the recogniser renders its own training images on the fly. Training: ensembles of MLPs, standardised
inputs and targets, mean squared error, Adam, vmapped over ensemble members (and over seeds in the rate sweep)."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .. import windows, models, sim, dynamics3d as dyn, sensors
from ..baseline import agent as bagent
from .agent import Norm, ErimParams


class Data(NamedTuple):
    feats: jnp.ndarray      # [n, (N+1)*FEAT]
    tx: jnp.ndarray
    tz: jnp.ndarray
    tf: jnp.ndarray
    tg: jnp.ndarray
    cls: jnp.ndarray        # [n]
    rel: jnp.ndarray        # [n, 3] Object position relative to the Machine (world)
    q_obj: jnp.ndarray      # [n, 4]
    light: jnp.ndarray      # [n, 3]
    y_obj: jnp.ndarray      # [n, 3] observed range/az/el (camera pointing)


def make_explore_agent(cfg, mcfg, sigma_u=0.5, p_random=1 / 3):
    """Baseline agent with exploration: with probability p_random (per mission, decided by the mission key) the
    controls are random; otherwise LQ + Gaussian noise. Keeps a window so that features are logged consistently."""
    B = bagent.bind(cfg)

    class Explore(B):
        @staticmethod
        def init(params, obs0, cfg_):
            y_own, y_obj, img = obs0
            S = B.init(params, obs0, cfg_)
            win = windows.init_window(mcfg.N, windows.row(y_own, y_obj, jnp.zeros(6)))
            return (S, win)

        @staticmethod
        def update(params, S, obs, u_prev, t):
            y_own, y_obj, img = obs
            S0, win = S
            S1, sm = B.update(params, S0, obs, u_prev, t)
            win = windows.push(win, windows.row(y_own, y_obj, u_prev))
            return (S1, win), sm

        @staticmethod
        def act(params, sm, obs, t, key):
            k1, k2 = jax.random.split(key)
            u_lq = B.act(params, sm, obs, t, key) + sigma_u * jax.random.normal(k1, (6,))
            u_rand = jax.random.normal(k2, (6,)) * jnp.array([cfg.u_max] * 3 + [cfg.w_max] * 3) * 0.7
            return jnp.where(params.random_flag, u_rand, u_lq)

    return Explore


class ExploreParams(NamedTuple):
    K: jnp.ndarray
    random_flag: jnp.ndarray


def generate(key, cfg, mcfg, n_missions, T, p_random=1 / 3):
    """Returns a Data pytree with n_missions * T rows (images are not stored; poses are)."""
    missions = dyn.sample_missions(key, n_missions, cfg)
    flags = jax.random.uniform(jax.random.fold_in(key, 99), (n_missions,)) < p_random
    base = bagent.make_params(cfg)
    params = ExploreParams(K=jnp.broadcast_to(base.K, (n_missions, 2)), random_flag=flags)
    Explore = make_explore_agent(cfg, mcfg)

    def extra(S, sm, obs, x, z, mission, t):
        _, win = S
        feats, anchor = windows.features(win)
        tx, tz, tf, tg = windows.targets(x, z, mission.f, mission.g, anchor)
        return (feats, tx, tz, tf, tg, mission.cls, z[0:3] - x[0:3], z[3:7], mission.light, obs[1])

    def one(m, p):
        _, _, _, ex = sim.rollout_one(Explore, p, m, cfg, T, 0.0, with_images=False, extra=extra)
        return ex

    ex = jax.vmap(one)(missions, params)
    flat = jax.tree_util.tree_map(lambda a: a.reshape((-1,) + a.shape[2:]), ex)
    return Data(*flat)


def norm_from(data):
    s = lambda a: jnp.maximum(jnp.std(a, 0), 1e-3)
    return Norm(jnp.mean(data.feats, 0), s(data.feats), jnp.mean(data.tx, 0), s(data.tx), jnp.mean(data.tz, 0), s(data.tz),
                jnp.mean(data.tf, 0), s(data.tf), jnp.mean(data.tg, 0), s(data.tg))


def train_ensemble(key, ens, X, Y, steps, bs, lr):
    """X, Y standardised. Each member sees its own minibatch stream."""
    E = jax.tree_util.tree_leaves(ens)[0].shape[0]
    n = X.shape[0]

    def loss(p, xb, yb):
        return jnp.mean((models.mlp(p, xb) - yb) ** 2)

    def member(p, k):
        st = models.adam_init(p)

        def step(carry, i):
            p, st = carry
            idx = jax.random.randint(jax.random.fold_in(k, i), (bs,), 0, n)
            l, g = jax.value_and_grad(loss)(p, X[idx], Y[idx])
            p, st = models.adam_update(p, g, st, models.cosine_lr(i, steps, lr), clip=1.0)
            return (p, st), l

        (p, _), ls = jax.lax.scan(step, (p, st), jnp.arange(steps))
        return p, ls[-50:].mean()

    return jax.vmap(member)(ens, jax.random.split(key, E))


def train_estimators(key, params, data, tcfg):
    """Returns params with trained est_* ensembles and a dict of final losses."""
    nm = params.norm
    X = (data.feats - nm.f_mean) / nm.f_std
    k = jax.random.split(key, 4)
    out, losses = {}, {}
    for name, ens, Y, mean, std, kk in (("est_x", params.est_x, data.tx, nm.tx_mean, nm.tx_std, k[0]),
                                        ("est_z", params.est_z, data.tz, nm.tz_mean, nm.tz_std, k[1]),
                                        ("est_f", params.est_f, data.tf, nm.tf_mean, nm.tf_std, k[2]),
                                        ("est_g", params.est_g, data.tg, nm.tg_mean, nm.tg_std, k[3])):
        p, l = train_ensemble(kk, ens, X, (Y - mean) / std, tcfg.est_steps, tcfg.est_bs, tcfg.est_lr)
        out[name], losses[name] = p, float(jnp.mean(l))
    return params._replace(**out), losses
