"""The classical agent: KF + EKF + template recogniser + LQ, under certainty equivalence."""
from typing import NamedTuple
import jax.numpy as jnp
from .. import render
from ..sim import Summary
from . import ekf, template, lq


class BaselineState(NamedTuple):
    kf: ekf.MachineKF
    oe: ekf.ObjectEKF
    log_post: jnp.ndarray


class BaselineParams(NamedTuple):
    K: jnp.ndarray          # LQ gain (2,)


def make_params(cfg):
    return BaselineParams(K=lq.dare_gain(cfg.dt, r=cfg.c_u))


class BaselineAgent:
    @staticmethod
    def init(params, obs0, cfg):
        y_own, y_obj, img = obs0
        kf = ekf.machine_init(y_own, cfg)
        oe = ekf.object_init(y_obj, kf.x[0:3], cfg)
        return BaselineState(kf, oe, jnp.full(render.N_CLASSES, -jnp.log(render.N_CLASSES)))

    @staticmethod
    def update(params, S, obs, u_prev, t):
        y_own, y_obj, img = obs
        cfg = BaselineAgent.cfg
        kf = ekf.machine_predict(S.kf, u_prev, cfg)
        kf, _, _ = ekf.machine_update(kf, y_own, cfg)
        oe = ekf.object_predict(S.oe, cfg)
        oe, _, _ = ekf.object_update(oe, y_obj, kf.x[0:3], cfg)
        rel_hat = oe.zeta[0:3] - kf.x[0:3]
        log_post = template.update_posterior(S.log_post, img, rel_hat, y_obj, cfg)
        pr = ekf.proxies(kf, oe)
        sm = Summary(x_hat=kf.x, z_hat=oe.zeta[0:6], f_hat=oe.zeta[6:10], g_hat=oe.zeta[10:12], log_post=log_post,
                     proxies=jnp.concatenate([pr, jnp.array([1.0 - jnp.exp(jnp.max(log_post))])]))
        return BaselineState(kf, oe, log_post), sm

    @staticmethod
    def act(params, sm, obs, t, key):
        return lq.lq_control(sm.x_hat, sm.z_hat, params.K, BaselineAgent.cfg)


def bind(cfg):
    """The agent functions need the SimCfg; it is attached as a class attribute to keep the Agent protocol uniform."""
    BaselineAgent.cfg = cfg
    return BaselineAgent
