"""The moving window of the ERIM estimators, defined once and used both to build training targets and online.

A window holds the last N+1 raw rows (y_own 13, y_obj 3, u_prev 6) = 22 numbers per row; the row written at stage t
carries the observations of stage t and the control applied at stage t-1 (zero at t = 0). Features (24 per row)
are computed relative to an anchor, the own-position observation of the newest row, so that they are translation
invariant: own p - anchor, attitude as the first two columns of R(q), v, w, Object relative Cartesian position from
(range, az, el), u_prev. Targets, built from the true states with the same anchor: x-part (p - anchor, R columns, v,
w) = 15, z-part (p_z - p_x, v_z) = 6, f = 4, g = 2."""
import jax.numpy as jnp
from . import quat, sensors

RAW = 22
FEAT = 24
DIM_TX, DIM_TZ, DIM_TF, DIM_TG = 15, 6, 4, 2


def att_cols(q):
    R = quat.to_rotmat(q)
    return jnp.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def row(y_own, y_obj, u_prev):
    return jnp.concatenate([y_own, y_obj, u_prev], -1)


def init_window(N, first_row):
    """At stage 0 the window is filled with the first row, so features are defined from the start."""
    return jnp.broadcast_to(first_row, (N + 1, RAW))


def push(win, new_row):
    return jnp.concatenate([win[1:], new_row[None]], 0)


def features(win):
    y_own, y_obj, u_prev = win[:, 0:13], win[:, 13:16], win[:, 16:22]
    anchor = y_own[-1, 0:3]
    p = y_own[:, 0:3] - anchor
    att = att_cols(y_own[:, 3:7])
    v, w = y_own[:, 7:10], y_own[:, 10:13]
    rel = sensors.rae_to_rel(y_obj) + (y_own[:, 0:3] - anchor)      # Object position relative to the anchor
    feats = jnp.concatenate([p, att, v, w, rel, u_prev], -1)
    return feats.reshape(-1), anchor


def targets(x, z, f, g, anchor):
    tx = jnp.concatenate([x[0:3] - anchor, att_cols(x[3:7]), x[7:10], x[10:13]], -1)
    tz = jnp.concatenate([z[0:3] - x[0:3], z[7:10]], -1)
    return tx, tz, f, g


def unpack_tx(tx, anchor):
    """x-estimate -> (p, R columns, v, w) in world terms."""
    return tx[0:3] + anchor, tx[3:9], tx[9:12], tx[12:15]
