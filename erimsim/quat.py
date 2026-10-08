"""Unit quaternions q = [w, x, y, z]; R(q) maps body to world. Error-state operations boxplus / boxminus on the
tangent (rotation vector), as used by the error-state Kalman filter and by the dynamics."""
import jax
import jax.numpy as jnp


def normalize(q):
    return q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def canonical(q):
    """Fix the sign ambiguity: w >= 0."""
    return q * jnp.where(q[..., :1] < 0, -1.0, 1.0)


def mul(a, b):
    aw, ax, ay, az = jnp.moveaxis(a, -1, 0)
    bw, bx, by, bz = jnp.moveaxis(b, -1, 0)
    return jnp.stack([aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw], -1)


def conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def to_rotmat(q):
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    r = jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
    return r


def _safe_norm(v):
    """sqrt(|v|^2 + tiny): equals |v| to float precision and has a finite derivative at v = 0 (the plain norm has a
    0/0 derivative there, which turns Kalman Jacobians into NaN at zero angular velocity)."""
    return jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + 1e-24)


def exp_map(theta):
    """Rotation vector -> unit quaternion (safe at zero, also for derivatives)."""
    a = _safe_norm(theta)
    half = 0.5 * a
    s = jnp.sin(half) / a
    return jnp.concatenate([jnp.cos(half), s * theta], -1)


def log_map(q):
    """Unit quaternion -> rotation vector (angle in (-pi, pi]), safe at the identity also for derivatives."""
    q = canonical(q)
    w = jnp.clip(q[..., :1], -1.0, 1.0)
    v = q[..., 1:]
    n = _safe_norm(v)
    angle = 2.0 * jnp.arctan2(n, w)
    return (angle / n) * v


def boxplus(q, dtheta):
    """q ⊕ δθ = q ⊗ exp(δθ)  (perturbation in the body frame)."""
    return normalize(mul(q, exp_map(dtheta)))


def boxminus(q1, q2):
    """q1 ⊖ q2 = log(q2^{-1} ⊗ q1): the body-frame rotation vector taking q2 to q1."""
    return log_map(mul(conj(q2), q1))


def rotate(q, v):
    return jnp.einsum('...ij,...j->...i', to_rotmat(q), v)


def random(key, shape=()):
    """Uniformly distributed unit quaternions."""
    u = jax.random.uniform(key, shape + (3,))
    a, b, c = u[..., 0], u[..., 1], u[..., 2]
    q = jnp.stack([jnp.sqrt(1 - a) * jnp.sin(2 * jnp.pi * b), jnp.sqrt(1 - a) * jnp.cos(2 * jnp.pi * b),
                   jnp.sqrt(a) * jnp.sin(2 * jnp.pi * c), jnp.sqrt(a) * jnp.cos(2 * jnp.pi * c)], -1)
    return canonical(q)


def identity(shape=()):
    return jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (4,))


def from_two_columns(c1, c2):
    """Orthonormalise two estimated columns of R (Gram-Schmidt) and return the unit quaternion (Shepperd's method,
    branch-free: the four candidates are weighted by softmax of their squared denominators)."""
    e1 = c1 / jnp.maximum(jnp.linalg.norm(c1, axis=-1, keepdims=True), 1e-9)
    c2 = c2 - jnp.sum(c2 * e1, -1, keepdims=True) * e1
    e2 = c2 / jnp.maximum(jnp.linalg.norm(c2, axis=-1, keepdims=True), 1e-9)
    e3 = jnp.cross(e1, e2)
    R = jnp.stack([e1, e2, e3], -1)          # columns
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q0 = jnp.stack([1 + tr, m21 - m12, m02 - m20, m10 - m01], -1)
    q1 = jnp.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1)
    q2 = jnp.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1)
    q3 = jnp.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1)
    cands = jnp.stack([q0, q1, q2, q3], -2)
    cands = cands / jnp.maximum(jnp.linalg.norm(cands, axis=-1, keepdims=True), 1e-9)
    # align signs with the first candidate's and pick the best-conditioned one smoothly
    sgn = jnp.sign(jnp.sum(cands * cands[..., :1, :], -1, keepdims=True) + 1e-12)
    cands = cands * sgn
    w = jax.nn.softmax(20.0 * jnp.stack([1 + tr, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1), -1)
    q = jnp.sum(w[..., :, None] * cands, -2)
    return canonical(normalize(q))
