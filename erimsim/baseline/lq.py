"""LQ tracking under certainty equivalence. The translational subsystem is a double integrator per axis; the gain
comes from the discrete algebraic Riccati equation of one axis (2 x 2), solved once by iteration. The desired world
acceleration is mapped to the body frame with the estimated attitude; the angular control damps the angular rate."""
import jax.numpy as jnp
import numpy as np
from .. import quat


def dare_gain(dt, q_pos=1.0, q_vel=0.1, r=0.1, iters=2000):
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    Q = np.diag([q_pos, q_vel])
    P = Q.copy()
    for _ in range(iters):
        K = np.linalg.solve(r + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    K = np.linalg.solve(r + B.T @ P @ B, B.T @ P @ A)
    return jnp.asarray(K[0])        # (2,): u = -(k_p e_p + k_v e_v) per axis


def lq_control(x_hat, z_hat, K, cfg, w_damp=2.0):
    p, q, v, w = x_hat[0:3], quat.normalize(x_hat[3:7]), x_hat[7:10], x_hat[10:13]
    pz, vz = z_hat[0:3], z_hat[3:6]
    pz_next = pz + cfg.dt * vz                                 # track the predicted Object position and velocity
    e_p, e_v = p - pz_next, v - vz
    a_world = -(K[0] * e_p + K[1] * e_v)
    a_body = quat.rotate(quat.conj(q), a_world)
    return jnp.concatenate([a_body, -w_damp * w])
