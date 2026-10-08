"""Training of the CNN recogniser (P_X). Images are rendered on the fly from the poses logged during the exploratory
missions (class, relative position, Object attitude, light, observed line of sight), with fresh pixel noise every
step, so no image dataset is ever stored. Cross-entropy loss, Adam."""
import jax
import jax.numpy as jnp
from .. import render, sensors, models


def render_batch(key, data, idx, cfg):
    def one(i, k):
        fwd, right, up = sensors.camera_axes(data.y_obj[i])
        return render.render(data.cls[i], data.rel[i], data.q_obj[i], fwd, right, up, data.light[i], k, cfg)

    return jax.vmap(one)(idx, jax.random.split(key, idx.shape[0]))


def train_cnn(key, cnn, data, cfg, tcfg, steps=None, bs=None, lr=None):
    steps = steps or tcfg.cnn_steps
    bs = bs or tcfg.cnn_bs
    lr = lr or tcfg.cnn_lr
    n = data.cls.shape[0]

    def loss(p, imgs, labels):
        logits = models.cnn(p, imgs)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels[:, None], 1))

    def step(carry, i):
        p, st = carry
        k = jax.random.fold_in(key, i)
        k1, k2 = jax.random.split(k)
        idx = jax.random.randint(k1, (bs,), 0, n)
        imgs = render_batch(k2, data, idx, cfg)
        l, g = jax.value_and_grad(loss)(p, imgs, data.cls[idx])
        p, st = models.adam_update(p, g, st, models.cosine_lr(i, steps, lr), clip=5.0)
        return (p, st), l

    (p, _), ls = jax.lax.scan(step, (cnn, models.adam_init(cnn)), jnp.arange(steps))
    return p, ls


def accuracy(key, cnn, data, cfg, n=512):
    idx = jax.random.randint(key, (min(n, data.cls.shape[0]),), 0, data.cls.shape[0])
    imgs = render_batch(jax.random.fold_in(key, 1), data, idx, cfg)
    pred = jnp.argmax(models.cnn(cnn, imgs), -1)
    return jnp.mean(pred == data.cls[idx])
