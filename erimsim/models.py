"""Fixed-structure parametrised (FSP) functions: multilayer perceptrons, a small strided CNN of depth J, ensembles
(a leading parameter axis handled with vmap), and a hand-rolled Adam so the package needs only JAX and numpy."""
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------- MLP
def init_mlp(key, sizes, scale=1.0):
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (a, b)) * scale / jnp.sqrt(a)
        params.append((w, jnp.zeros(b)))
    return params


def mlp(params, x, act=jnp.tanh):
    for w, b in params[:-1]:
        x = act(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def n_params(params):
    return sum(int(p.size) for p in jax.tree_util.tree_leaves(params))


# ---------------------------------------------------------------- CNN
def init_cnn(key, J, ch, fc, n_out, in_ch=1):
    """J strided 3x3 convolutions (channels ch[:J]), ReLU, global average pooling, one dense layer, logits."""
    ch = tuple(ch)[:J] + (ch[-1],) * max(0, J - len(ch))
    convs = []
    c_in = in_ch
    for c_out in ch:
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (3, 3, c_in, c_out)) * jnp.sqrt(2.0 / (9 * c_in))
        convs.append((w, jnp.zeros(c_out)))
        c_in = c_out
    key, k1, k2 = jax.random.split(key, 3)
    w1 = jax.random.normal(k1, (c_in, fc)) * jnp.sqrt(2.0 / c_in)
    w2 = jax.random.normal(k2, (fc, n_out)) / jnp.sqrt(fc)
    return {"convs": convs, "fc": (w1, jnp.zeros(fc)), "out": (w2, jnp.zeros(n_out))}


def cnn(params, img):
    """img: [H, W] or [B, H, W] -> logits [n_out] or [B, n_out]."""
    single = img.ndim == 2
    x = img[None] if single else img
    x = x[..., None]                                                   # NHWC
    for w, b in params["convs"]:
        x = jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b
        x = jax.nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))
    w1, b1 = params["fc"]
    x = jax.nn.relu(x @ w1 + b1)
    w2, b2 = params["out"]
    out = x @ w2 + b2
    return out[0] if single else out


# ---------------------------------------------------------------- ensembles
def init_ensemble(key, E, init_fn):
    return jax.vmap(init_fn)(jax.random.split(key, E))


def ensemble_apply(apply_fn, params, x):
    """params with a leading ensemble axis; x shared. Returns [E, ...]."""
    return jax.vmap(apply_fn, in_axes=(0, None))(params, x)


# ---------------------------------------------------------------- Adam
def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": z, "v": z, "t": jnp.zeros(())}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, clip=None):
    if clip is not None:
        gn = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, clip / (gn + 1e-12)), grads)
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, state["v"], grads)
    mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
    new = jax.tree_util.tree_map(lambda p, a, c: p - lr * a / (jnp.sqrt(c) + eps), params, mh, vh)
    return new, {"m": m, "v": v, "t": t}


def cosine_lr(step, steps, lr, warmup=100):
    w = jnp.minimum(1.0, (step + 1) / warmup)
    c = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step / steps, 1.0)))
    return lr * w * jnp.maximum(c, 0.05)
