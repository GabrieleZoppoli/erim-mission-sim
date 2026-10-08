"""Image formation. The Object is one of eight solids of nominal radius 1 m, described by signed distance functions
(SDF) in its body frame. A pinhole camera at the Machine looks along the observed line of sight; each pixel's ray is
sphere-traced for a fixed number of steps (branch-free, hence vmap/jit friendly and differentiable with respect to
the relative pose); hits are Lambert-shaded with a per-mission light direction; then Gaussian blur and pixel noise,
both growing with range. Convention asserted by tests/test_render.py: images are [row, col], row 0 at the top,
column index grows to the camera's right; apparent radius of a unit sphere = k_focal * img / range pixels."""
import jax
import jax.numpy as jnp
from jax import lax
from . import quat

N_CLASSES = 8
CLASS_NAMES = ("sphere", "cube", "cylinder", "cone", "torus", "capsule", "bracket", "cross")


def _norm(v):
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + 1e-12)


def sd_box(p, half):
    q = jnp.abs(p) - half
    return _norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0)


def sd_sphere(p):
    return _norm(p) - 1.0


def sd_cube(p):
    return sd_box(p, jnp.array([0.8, 0.8, 0.8]))


def sd_cylinder(p):
    d = jnp.stack([_norm(p[..., :2]) - 0.8, jnp.abs(p[..., 2]) - 1.0], -1)
    return jnp.minimum(jnp.max(d, axis=-1), 0.0) + _norm(jnp.maximum(d, 0.0))


def sd_cone(p):
    """Apex at z = +1, base of radius 1 at z = -1 (exact capped cone, after I. Quilez)."""
    q = jnp.array([1.0, -2.0])                       # (base radius, -height)
    w = jnp.stack([_norm(p[..., :2]), p[..., 2] - 1.0], -1)
    a = w - q * jnp.clip(jnp.sum(w * q, -1, keepdims=True) / jnp.sum(q * q), 0.0, 1.0)
    b = w - q * jnp.stack([jnp.clip(w[..., 0] / q[0], 0.0, 1.0), jnp.ones_like(w[..., 0])], -1)
    k = jnp.sign(q[1])
    d = jnp.minimum(jnp.sum(a * a, -1), jnp.sum(b * b, -1))
    s = jnp.maximum(k * (w[..., 0] * q[1] - w[..., 1] * q[0]), k * (w[..., 1] - q[1]))
    return jnp.sqrt(d + 1e-12) * jnp.sign(s)        # +1e-12: finite gradient on the surface


def sd_torus(p):
    q = jnp.stack([_norm(p[..., :2]) - 0.75, p[..., 2]], -1)
    return _norm(q) - 0.3


def sd_capsule(p):
    a, b, r = jnp.array([0.0, 0.0, -0.7]), jnp.array([0.0, 0.0, 0.7]), 0.5
    pa, ba = p - a, b - a
    h = jnp.clip(jnp.sum(pa * ba, -1, keepdims=True) / jnp.sum(ba * ba), 0.0, 1.0)
    return _norm(pa - ba * h) - r


def sd_bracket(p):
    return jnp.minimum(sd_box(p - jnp.array([0.0, 0.0, -0.6]), jnp.array([0.9, 0.3, 0.3])),
                       sd_box(p - jnp.array([-0.6, 0.0, 0.0]), jnp.array([0.3, 0.3, 0.9])))


def sd_cross(p):
    return jnp.minimum(jnp.minimum(sd_box(p, jnp.array([1.0, 0.25, 0.25])), sd_box(p, jnp.array([0.25, 1.0, 0.25]))),
                       sd_box(p, jnp.array([0.25, 0.25, 1.0])))


PRIMITIVES = (sd_sphere, sd_cube, sd_cylinder, sd_cone, sd_torus, sd_capsule, sd_bracket, sd_cross)
HIT_EPS = 0.03


def sdf_oh(oh, p):
    """Signed distance of points p (..., 3) in the Object frame; oh = one-hot class weights (8,), so that the class
    is an ordinary array argument (vmappable, custom-vjp friendly)."""
    allsd = jnp.stack([f(p) for f in PRIMITIVES], 0)          # (8, ...)
    return jnp.tensordot(oh, allsd, axes=(0, 0))


def sdf(cls, p):
    return sdf_oh(jax.nn.one_hot(cls, N_CLASSES), p)


def make_march(steps):
    """Sphere tracing with a custom backward pass: forward runs the fixed-step march under lax.scan (no gradient
    tracking); backward differentiates the hit distance by the implicit function theorem, s(o + t d) = 0 =>
    dt/do = -grad s / (grad s . d), dt/dd = -t grad s / (grad s . d), zero for rays that miss. This avoids both the
    cost of differentiating through the march and a NaN produced by scan's transpose in some JAX versions."""

    def _forward(o, d, t0, t_max, oh):
        t = jnp.full(d.shape[:-1], t0)

        def body(t, _):
            s = sdf_oh(oh, o + t[..., None] * d)
            return jnp.minimum(t + jnp.maximum(s, 0.0), t_max), None

        t, _ = lax.scan(body, t, None, length=steps)
        return t

    @jax.custom_vjp
    def march(o, d, t0, t_max, oh):
        return _forward(o, d, t0, t_max, oh)

    def fwd(o, d, t0, t_max, oh):
        t = _forward(o, d, t0, t_max, oh)
        return t, (o, d, t, t_max, oh)

    def bwd(res, ct):
        o, d, t, t_max, oh = res
        p = o + t[..., None] * d
        g = jax.vmap(jax.vmap(jax.grad(lambda q: sdf_oh(oh, q))))(p)
        s = sdf_oh(oh, p)
        hit = (s < HIT_EPS) & (t < t_max - 1e-4)
        denom = jnp.sum(g * d, -1)
        denom = jnp.where(jnp.abs(denom) < 1e-3, jnp.where(denom < 0, -1e-3, 1e-3), denom)
        dt_do = jnp.where(hit[..., None], -g / denom[..., None], 0.0)
        dt_dd = jnp.where(hit[..., None], -t[..., None] * g / denom[..., None], 0.0)
        return (jnp.sum(ct[..., None] * dt_do, axis=(0, 1)), ct[..., None] * dt_dd,
                jnp.zeros(()), jnp.zeros(()), jnp.zeros_like(oh))

    march.defvjp(fwd, bwd)
    return march


_MARCH = {}


def march_fn(steps):
    if steps not in _MARCH:
        _MARCH[steps] = make_march(steps)
    return _MARCH[steps]


def pixel_rays(cfg, fwd, right, up):
    H = W = cfg.img
    f = cfg.k_focal * cfg.img
    ii, jj = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    xs = (jj + 0.5 - W / 2) / f
    ys = -(ii + 0.5 - H / 2) / f
    d = fwd[None, None, :] + xs[..., None] * right[None, None, :] + ys[..., None] * up[None, None, :]
    return d / _norm(d)[..., None]


def render_clean(cls, rel_pos, q_obj, fwd, right, up, light, cfg):
    """Noise-free shaded image [H, W] in [0, 1]. rel_pos: Object position relative to the camera (world frame)."""
    dirs = pixel_rays(cfg, fwd, right, up)
    Rt = quat.to_rotmat(q_obj).T                                  # world -> body
    o = Rt @ (-rel_pos)                                           # camera position in the Object frame
    d = jnp.einsum("ij,hwj->hwi", Rt, dirs)
    t0 = jnp.maximum(_norm(o) - 2.0, 0.0)
    t_max = _norm(o) + 4.0                                        # beyond this the ray has passed the Object
    oh = jax.nn.one_hot(cls, N_CLASSES)
    t = march_fn(cfg.march_steps)(o, d, t0, t_max, oh)
    p = o + t[..., None] * d
    s = sdf_oh(oh, p)
    hit = s < HIT_EPS
    e = 1e-3
    ex, ey, ez = jnp.array([e, 0, 0]), jnp.array([0, e, 0]), jnp.array([0, 0, e])
    n = jnp.stack([sdf_oh(oh, p + ex) - sdf_oh(oh, p - ex), sdf_oh(oh, p + ey) - sdf_oh(oh, p - ey),
                   sdf_oh(oh, p + ez) - sdf_oh(oh, p - ez)], -1)
    n = n / _norm(n)[..., None]
    n_world = jnp.einsum("ij,hwj->hwi", Rt.T, n)
    lam = jnp.maximum(jnp.sum(n_world * light, -1), 0.0)
    shade = 0.25 + 0.75 * lam
    return jnp.where(hit, shade, 0.0)


def gaussian_blur(img, sigma, taps=7):
    k = jnp.arange(taps) - taps // 2
    w = jnp.exp(-0.5 * (k / jnp.maximum(sigma, 1e-3)) ** 2)
    w = w / jnp.sum(w)
    pad = taps // 2
    x = jnp.pad(img, ((pad, pad), (0, 0)))
    x = sum(w[i] * x[i:i + img.shape[0], :] for i in range(taps))
    x = jnp.pad(x, ((0, 0), (pad, pad)))
    x = sum(w[i] * x[:, i:i + img.shape[1]] for i in range(taps))
    return x


def render(cls, rel_pos, q_obj, fwd, right, up, light, key, cfg):
    """Observed image: clean render, blur sd = blur0 + blur_r * r, pixel noise sd = sd_px0 + sd_px_r * r / r0."""
    r = _norm(rel_pos)
    img = render_clean(cls, rel_pos, q_obj, fwd, right, up, light, cfg)
    img = gaussian_blur(img, cfg.blur0 + cfg.blur_r * r)
    noise = jax.random.normal(key, img.shape) * (cfg.sd_px0 + cfg.sd_px_r * r / cfg.r0)
    return jnp.clip(img + noise, 0.0, 1.0)


def silhouette(img, thr=0.15):
    return (img > thr).astype(jnp.float32)
