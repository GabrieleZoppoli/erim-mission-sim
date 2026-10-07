import os, tempfile
import jax, jax.numpy as jnp, numpy as np
from erimsim import models, windows, io, quat, dynamics3d as dyn, sensors
from erimsim.config import SimCfg


def test_mlp_cnn_shapes_and_ensemble():
    k = jax.random.PRNGKey(0)
    p = models.init_mlp(k, [24, 32, 5])
    assert models.mlp(p, jnp.ones(24)).shape == (5,)
    assert models.mlp(p, jnp.ones((7, 24))).shape == (7, 5)
    c = models.init_cnn(k, J=3, ch=(8, 16, 16), fc=16, n_out=8)
    assert models.cnn(c, jnp.zeros((16, 16))).shape == (8,)
    assert models.cnn(c, jnp.zeros((4, 64, 64))).shape == (4, 8)
    ens = models.init_ensemble(k, 3, lambda kk: models.init_mlp(kk, [24, 16, 5]))
    out = models.ensemble_apply(models.mlp, ens, jnp.ones(24))
    assert out.shape == (3, 5)
    assert models.n_params(p) == 24 * 32 + 32 + 32 * 5 + 5


def test_adam_minimises_quadratic():
    p = {"x": jnp.array([3.0, -2.0])}
    st = models.adam_init(p)
    loss = lambda q: jnp.sum((q["x"] - 1.0) ** 2)
    for _ in range(300):
        g = jax.grad(loss)(p)
        p, st = models.adam_update(p, g, st, lr=0.05, clip=10.0)
    assert float(loss(p)) < 1e-3


def test_window_alignment_and_targets():
    cfg = SimCfg()
    m = dyn.sample_mission(jax.random.PRNGKey(1), cfg)
    x, z = m.x0, m.z0
    rows, us = [], []
    win = None
    u_prev = jnp.zeros(6)
    for t in range(4):
        y_own = sensors.observe_own(x, jnp.zeros(12), cfg)
        y_obj = sensors.observe_object(x, z, m.g, jnp.zeros(3), cfg)
        r = windows.row(y_own, y_obj, u_prev)
        win = windows.init_window(2, r) if win is None else windows.push(win, r)
        u = jnp.array([0.5, 0, 0, 0, 0, 0.1]) * (t + 1)
        x = dyn.machine_step(x, u, jnp.zeros(6), cfg)
        z = dyn.object_step(z, m.f, jnp.zeros(6), cfg)
        rows.append(r); us.append(u); u_prev = u
    # the newest row carries u applied at the previous stage, i.e. us[2] after the 4th push (rows 1..3 kept)
    np.testing.assert_allclose(win[-1, 16:22], us[2])
    np.testing.assert_allclose(win[-2, 16:22], us[1])
    feats, anchor = windows.features(win)
    assert feats.shape == (3 * windows.FEAT,)
    np.testing.assert_allclose(anchor, rows[3][0:3])
    tx, tz, tf, tg = windows.targets(x, z, m.f, m.g, anchor)
    assert tx.shape == (15,) and tz.shape == (6,) and tf.shape == (4,) and tg.shape == (2,)
    p, _, _, _ = windows.unpack_tx(tx, anchor)
    np.testing.assert_allclose(p, x[0:3], atol=1e-6)
    # the relative-position feature of the newest row equals the true relative position (noise-free observation, g=(0,1))
    y_obj0 = sensors.observe_object(x, z, jnp.array([0.0, 1.0]), jnp.zeros(3), cfg)
    np.testing.assert_allclose(sensors.rae_to_rel(y_obj0), z[0:3] - x[0:3], atol=1e-5)


def test_io_roundtrip_and_csv():
    tree = {"a": jnp.arange(3.0), "b": [jnp.ones((2, 2)), jnp.array(4.0)]}
    with tempfile.TemporaryDirectory() as d:
        io.save_pytree(os.path.join(d, "t.npz"), tree)
        back = io.load_pytree(os.path.join(d, "t.npz"), tree)
        np.testing.assert_allclose(back["b"][0], tree["b"][0])
        io.append_csv(os.path.join(d, "r.csv"), [{"x": 1, "y": 2.5}, {"x": 2, "y": float("nan")}])
        io.append_csv(os.path.join(d, "r.csv"), [{"x": 3, "y": None}])
        lines = open(os.path.join(d, "r.csv")).read().strip().splitlines()
        assert lines[0] == "x,y" and len(lines) == 4 and lines[2].endswith("NA") and lines[3].endswith("NA")
        io.mark_done(d, "E2/n8_M50_s0"); assert io.is_done(d, "E2/n8_M50_s0") and not io.is_done(d, "E2/n8_M50_s1")
