import jax, jax.numpy as jnp, numpy as np
from erimsim import render, quat, sensors
from erimsim.config import SimCfg

cfg = SimCfg(img=64, march_steps=32)
light = jnp.array([0.0, 0.0, 1.0])
q_id = quat.identity()


def axes_towards(rel):
    y = sensors.rel_to_rae(rel)
    return sensors.camera_axes(y)


def area(img):
    return float(jnp.sum(img > 0.05))


def centroid(img):
    ii, jj = jnp.meshgrid(jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing="ij")
    m = (img > 0.05).astype(jnp.float32)
    return float(jnp.sum(ii * m) / jnp.sum(m)), float(jnp.sum(jj * m) / jnp.sum(m))


def test_sphere_apparent_area_and_scale():
    rel = jnp.array([8.0, 0.0, 0.0])
    fwd, right, up = axes_towards(rel)
    img8 = render.render_clean(0, rel, q_id, fwd, right, up, light, cfg)
    r_px = cfg.k_focal * cfg.img / 8.0
    assert abs(area(img8) - np.pi * r_px ** 2) / (np.pi * r_px ** 2) < 0.15
    img16 = render.render_clean(0, jnp.array([16.0, 0.0, 0.0]), q_id, fwd, right, up, light, cfg)
    assert abs(area(img16) / area(img8) - 0.25) < 0.08


def test_orientation_right_is_higher_column_up_is_lower_row():
    rel = jnp.array([8.0, 0.0, 0.0])
    fwd, right, up = axes_towards(rel)
    c0 = centroid(render.render_clean(0, rel, q_id, fwd, right, up, light, cfg))
    c_right = centroid(render.render_clean(0, rel + 1.5 * right, q_id, fwd, right, up, light, cfg))
    c_up = centroid(render.render_clean(0, rel + 1.5 * up, q_id, fwd, right, up, light, cfg))
    assert c_right[1] > c0[1] + 3
    assert c_up[0] < c0[0] - 3
    assert abs(c0[0] - cfg.img / 2) < 1.5 and abs(c0[1] - cfg.img / 2) < 1.5


def test_all_classes_visible_and_distinct():
    rel = jnp.array([6.0, 0.0, 0.0])
    fwd, right, up = axes_towards(rel)
    q = quat.exp_map(jnp.array([0.4, 0.3, 0.2]))
    imgs = [render.render_clean(c, rel, q, fwd, right, up, light, cfg) for c in range(8)]
    for im in imgs:
        assert 50 < area(im) < cfg.img * cfg.img * 0.6
    sil = [render.silhouette(im) for im in imgs]
    for a in range(8):
        for b in range(a + 1, 8):
            assert float(jnp.mean(jnp.abs(sil[a] - sil[b]))) > 0.01


def test_gradient_wrt_relative_position_is_finite():
    rel = jnp.array([8.0, 0.5, -0.3])
    fwd, right, up = axes_towards(rel)
    for c in (3, 6):
        f = lambda r: jnp.sum(render.render_clean(c, r, q_id, fwd, right, up, light, cfg))
        g = jax.grad(f)(rel)
        assert bool(jnp.all(jnp.isfinite(g)))
    g = jax.grad(lambda r: jnp.sum(render.render_clean(0, r, q_id, fwd, right, up, light, cfg)))(rel)
    assert float(jnp.linalg.norm(g)) > 0        # the image does depend on the relative position


def test_march_gradient_matches_analytic_sphere():
    """For a unit sphere centred at c = -o (camera at o in the object frame) and a ray o + t d, the hit distance is
    t = -(o.d) - sqrt((o.d)^2 - |o|^2 + 1); its gradient wrt o is (-d - (o - (o.d) d)/sqrt(...))."""
    rel = jnp.array([6.0, 0.4, 0.2])
    fwd, right, up = axes_towards(rel)
    dirs = render.pixel_rays(cfg, fwd, right, up)
    o = -rel
    march = render.march_fn(cfg.march_steps)
    oh = jax.nn.one_hot(0, render.N_CLASSES)
    t0, t_max = jnp.maximum(render._norm(o) - 2.0, 0.0), render._norm(o) + 4.0
    J = jax.jacrev(lambda oo: march(oo, dirs, t0, t_max, oh))(o)          # [H, W, 3]
    od = jnp.sum(dirs * o, -1)
    disc = od ** 2 - jnp.sum(o * o) + 1.0
    t_num = march(o, dirs, t0, t_max, oh)
    s_num = render.sdf_oh(oh, o + t_num[..., None] * dirs)
    interior = (disc > 0.3) & (s_num < render.HIT_EPS)          # away from the limb, where grazing rays converge slowly
    miss = disc < -0.05
    t_an = -od - jnp.sqrt(jnp.where(disc > 0, disc, 1.0))
    # d/do [-(o.d) - sqrt(D)] = -d - ((o.d) d - o) / sqrt(D) = -d + (o - (o.d) d) / sqrt(D)
    dt_an = -dirs + (o[None, None, :] - od[..., None] * dirs) / jnp.sqrt(jnp.where(disc > 0, disc, 1.0))[..., None]
    assert int(interior.sum()) > 50
    assert float(jnp.max(jnp.abs(jnp.where(interior, t_num - t_an, 0.0)))) < 0.05
    err = jnp.where(interior[..., None], J - dt_an, 0.0)
    assert float(jnp.max(jnp.abs(err))) < 0.1
    assert float(jnp.max(jnp.abs(jnp.where(miss[..., None], J, 0.0)))) == 0.0


def test_noisy_render_in_range_and_rowmajor_shape():
    rel = jnp.array([10.0, 2.0, 1.0])
    fwd, right, up = axes_towards(rel)
    img = render.render(5, rel, q_id, fwd, right, up, light, jax.random.PRNGKey(0), cfg)
    assert img.shape == (cfg.img, cfg.img)
    assert float(img.min()) >= 0 and float(img.max()) <= 1
