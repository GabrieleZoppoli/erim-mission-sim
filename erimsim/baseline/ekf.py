"""Estimators of the classical chain.

Machine: error-state Kalman filter. Nominal state x = (p, q, v, w) propagated with the known control; error state
(δp, δθ, δv, δw) in R^12; full-state measurement y_own with the attitude residual yq ⊖ q. Joseph-form update.

Object: extended Kalman filter on ζ = (p_z 3, v_z 3, om 3, c_d, b, s) in R^12. The dynamics of (p_z, v_z) are those
of dynamics3d.object_step (world-frame velocity rotating about om and decaying with c_d); om, c_d, b, s are constant
with a small random-walk floor so they remain identifiable. Measurement: y_obj = (s r + b, az, el) with r, az, el
computed from p_z - p_x_hat (the Machine's estimate is treated as known). Jacobians by automatic differentiation."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .. import quat, sensors


class MachineKF(NamedTuple):
    x: jnp.ndarray      # (13,) nominal state
    P: jnp.ndarray      # (12, 12)


class ObjectEKF(NamedTuple):
    zeta: jnp.ndarray   # (12,)
    P: jnp.ndarray      # (12, 12)


def skew(a):
    return jnp.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


# ---------------------------------------------------------------- Machine
def machine_init(y_own0, cfg):
    x = jnp.concatenate([y_own0[0:3], quat.normalize(y_own0[3:7]), y_own0[7:10], y_own0[10:13]])
    P = jnp.diag(jnp.concatenate([jnp.full(3, cfg.sd_yp ** 2), jnp.full(3, cfg.sd_yq ** 2),
                                  jnp.full(3, cfg.sd_yv ** 2), jnp.full(3, cfg.sd_yw ** 2)]))
    return MachineKF(x, P)


def machine_predict(kf, u, cfg):
    p, q, v, w = kf.x[0:3], kf.x[3:7], kf.x[7:10], kf.x[10:13]
    R = quat.to_rotmat(q)
    a_body = u[0:3]
    v1 = v + cfg.dt * (R @ a_body)
    p1 = p + cfg.dt * v1
    w1 = w + cfg.dt * u[3:6]
    q1 = quat.boxplus(q, cfg.dt * w1)
    I3, Z3 = jnp.eye(3), jnp.zeros((3, 3))
    A = jnp.block([[Z3, Z3, I3, Z3], [Z3, Z3, Z3, I3], [Z3, -R @ skew(a_body), Z3, Z3], [Z3, Z3, Z3, Z3]])
    F = jnp.eye(12) + cfg.dt * A
    Q = jnp.diag(jnp.concatenate([jnp.full(3, 1e-8), jnp.full(3, 1e-8), jnp.full(3, cfg.sd_v ** 2), jnp.full(3, cfg.sd_w ** 2)]))
    P = F @ kf.P @ F.T + Q
    return MachineKF(jnp.concatenate([p1, q1, v1, w1]), P)


def machine_update(kf, y_own, cfg):
    p, q, v, w = kf.x[0:3], kf.x[3:7], kf.x[7:10], kf.x[10:13]
    r = jnp.concatenate([y_own[0:3] - p, quat.boxminus(quat.normalize(y_own[3:7]), q), y_own[7:10] - v, y_own[10:13] - w])
    Rm = jnp.diag(jnp.concatenate([jnp.full(3, cfg.sd_yp ** 2), jnp.full(3, cfg.sd_yq ** 2),
                                   jnp.full(3, cfg.sd_yv ** 2), jnp.full(3, cfg.sd_yw ** 2)]))
    S = kf.P + Rm
    K = jnp.linalg.solve(S.T, kf.P.T).T                     # P S^{-1}
    d = K @ r
    x = jnp.concatenate([p + d[0:3], quat.boxplus(q, d[3:6]), v + d[6:9], w + d[9:12]])
    IK = jnp.eye(12) - K
    P = IK @ kf.P @ IK.T + K @ Rm @ K.T
    return MachineKF(x, 0.5 * (P + P.T)), r, S


# ---------------------------------------------------------------- Object
def object_f(zeta, cfg):
    p, v, om, cd = zeta[0:3], zeta[3:6], zeta[6:9], zeta[9]
    Rom = quat.to_rotmat(quat.exp_map(cfg.dt * om))
    v1 = (1.0 - cd * cfg.dt) * (Rom @ v)
    p1 = p + cfg.dt * v1
    return jnp.concatenate([p1, v1, zeta[6:12]])


def object_h(zeta, p_x):
    d = zeta[0:3] - p_x
    rae = sensors.rel_to_rae(d)
    return jnp.array([zeta[11] * rae[0] + zeta[10], rae[1], rae[2]])


def object_init(y_obj0, p_x0, cfg):
    rel = sensors.rae_to_rel(y_obj0)
    zeta = jnp.concatenate([p_x0 + rel, jnp.zeros(3), jnp.zeros(3), jnp.array([0.5 * cfg.cd_max, 0.0, 1.0])])
    P = jnp.diag(jnp.concatenate([jnp.full(3, 1.0), jnp.full(3, cfg.sd_vz0 ** 2 * 2), jnp.full(3, cfg.om_max ** 2),
                                  jnp.array([(0.5 * cfg.cd_max) ** 2, cfg.b_max ** 2, cfg.s_dev ** 2])]))
    return ObjectEKF(zeta, P)


def object_predict(ekf, cfg):
    F = jax.jacfwd(lambda z: object_f(z, cfg))(ekf.zeta)
    Q = jnp.diag(jnp.concatenate([jnp.full(3, 1e-8), jnp.full(3, cfg.sd_vz ** 2), jnp.full(3, 1e-7),
                                  jnp.array([1e-8, 1e-8, 1e-9])]))
    return ObjectEKF(object_f(ekf.zeta, cfg), F @ ekf.P @ F.T + Q)


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def object_update(ekf, y_obj, p_x, cfg):
    h = object_h(ekf.zeta, p_x)
    H = jax.jacfwd(lambda z: object_h(z, p_x))(ekf.zeta)
    r = y_obj - h
    r = r.at[1].set(wrap(r[1]))
    grow = 1.0 + jnp.maximum(h[0], 0.5) / cfg.r0
    Rm = jnp.diag(jnp.array([(cfg.sd_r0 * grow) ** 2, (cfg.sd_a0 * grow) ** 2, (cfg.sd_a0 * grow) ** 2]))
    S = H @ ekf.P @ H.T + Rm
    K = jnp.linalg.solve(S.T, (ekf.P @ H.T).T).T
    zeta = ekf.zeta + K @ r
    lo = jnp.array([-jnp.inf] * 6 + [-1.5 * cfg.om_max] * 3 + [0.0, -2 * cfg.b_max, 1 - 3 * cfg.s_dev])
    hi = jnp.array([jnp.inf] * 6 + [1.5 * cfg.om_max] * 3 + [2.5 * cfg.cd_max, 2 * cfg.b_max, 1 + 3 * cfg.s_dev])
    zeta = jnp.clip(zeta, lo, hi)
    IK = jnp.eye(12) - K @ H
    P = IK @ ekf.P @ IK.T + K @ Rm @ K.T
    P = 0.5 * (P + P.T) + 1e-10 * jnp.eye(12)
    return ObjectEKF(zeta, P), r, S


def proxies(kf, ekf):
    """Covariance-trace proxies: x (p, v), z (p_z, v_z), f (om, c_d), g (b, s)."""
    Px = kf.P
    return jnp.array([jnp.trace(Px[0:3, 0:3]) + jnp.trace(Px[6:9, 6:9]), jnp.trace(ekf.P[0:6, 0:6]),
                      jnp.trace(ekf.P[6:10, 6:10]), jnp.trace(ekf.P[10:12, 10:12])])
