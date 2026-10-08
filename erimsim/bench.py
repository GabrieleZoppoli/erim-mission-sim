"""Benchmark used to size the GPU runs: python -m erimsim.bench [--gpu N]. Prints the device and the wall-clock time
of the building blocks at full size, so that the run sheet's estimates can be checked before launching anything long."""
import argparse
import os
import sys
import time


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default=None)
    a = p.parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax, jax.numpy as jnp
    from . import io, sim, dynamics3d as dyn, render, sensors
    from .config import Cfg
    from .baseline import agent as bagent
    from .erim import agent as eagent, estimators, policy
    c = Cfg(); cfg, mcfg, tcfg = c.sim, c.model, c.train
    print("device:", io.device_info(), "python", sys.version.split()[0], flush=True)

    def timed(name, f, *args):
        t0 = time.time(); r = f(*args); jax.block_until_ready(r); t1 = time.time()
        r = f(*args); jax.block_until_ready(r); t2 = time.time()
        print(f"{name:48s} compile+run {t1 - t0:7.1f}s   run {t2 - t1:7.2f}s", flush=True)
        return r

    ms = dyn.sample_missions(jax.random.PRNGKey(0), 256, cfg)
    rel = ms.z0[:, 0:3] - ms.x0[:, 0:3]
    y = jax.vmap(sensors.rel_to_rae)(rel)
    fr = jax.vmap(sensors.camera_axes)(y)
    f_render = jax.jit(jax.vmap(lambda c_, r_, q_, f_, ri, u_, l_, k_: render.render(c_, r_, q_, f_, ri, u_, l_, k_, cfg)))
    timed("render 256 images 64x64", f_render, ms.cls, rel, ms.z0[:, 3:7], fr[0], fr[1], fr[2], ms.light, jax.random.split(jax.random.PRNGKey(1), 256))
    B = bagent.bind(cfg); bp = bagent.make_params(cfg)
    timed("baseline rollout 256 missions x 100 stages", jax.jit(lambda m: sim.rollout(B, bp, m, cfg, 100)[2]), ms)
    data = estimators.generate(jax.random.PRNGKey(2), cfg, mcfg, 8, 20)
    params = eagent.init_params(jax.random.PRNGKey(3), cfg, mcfg, estimators.norm_from(data))
    A = eagent.bind(cfg, mcfg, use_images=True, use_post=False)
    timed("ERIM rollout 256 missions x 100 stages (images)", jax.jit(lambda m: sim.rollout(A, params, m, cfg, 100)[2]), ms)
    A0 = eagent.bind(cfg, mcfg, use_images=False, use_post=False)
    g = jax.jit(jax.grad(lambda pol, m: jnp.mean(sim.rollout(A0, params._replace(pol=pol), m, cfg, 100, 0.0, False)[2])))
    timed("policy-search gradient, batch 256 x 100 stages", g, params.pol, ms)
    A1 = eagent.bind(cfg, mcfg, use_images=True, use_post=True)
    ms128 = jax.tree_util.tree_map(lambda a_: a_[:128], ms)
    g1 = jax.jit(jax.grad(lambda pol, m: jnp.mean(sim.rollout(A1, params._replace(pol=pol), m, cfg, 100, 0.1, True)[2])))
    timed("dual policy gradient (images in loop), batch 128 x 100", g1, params.pol, ms128)
    X = jnp.zeros((4096, (mcfg.N + 1) * 24)); Y = jnp.zeros((4096, 15))
    timed("estimator ensemble training, 200 steps", jax.jit(lambda k: estimators.train_ensemble(k, params.est_x, X, Y, 200, 256, 1e-3)[0]), jax.random.PRNGKey(4))
    print("scale: E1 eval = 1000 missions x 300 stages ~ 12 x the 256x100 rollout; policy search = pol_steps gradients;", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
