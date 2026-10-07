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


def exp_map(theta):
    """Rotation vector -> unit quaternion (safe at zero)."""
    a = jnp.linalg.norm(theta, axis=-1, keepdims=True)
    half = 0.5 * a
    s = jnp.where(a > 1e-8, jnp.sin(half) / jnp.maximum(a, 1e-12), 0.5 - a * a / 48.0)
    return jnp.concatenate([jnp.cos(half), s * theta], -1)


def log_map(q):
    """Unit quaternion -> rotation vector (angle in (-pi, pi])."""
    q = canonical(q)
    w = jnp.clip(q[..., :1], -1.0, 1.0)
    v = q[..., 1:]
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * jnp.arctan2(n, w)
    k = jnp.where(n > 1e-8, angle / jnp.maximum(n, 1e-12), 2.0 / jnp.maximum(w, 1e-12))
    return k * v


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
