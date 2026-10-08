"""Command line: python -m erimsim --exp E1 --out DIR [--smoke] [--gpu N] [--mem-fraction F] [--resume]
[--shard k/n] [--units GLOB] [--seeds 0,1,2]. Everything is written under --out/<exp>/; a unit that completed leaves
a marker and is skipped with --resume. GPU selection and memory flags are set before JAX is imported."""
import argparse
import fnmatch
import os
import sys
import time
import traceback


def parse(argv=None):
    p = argparse.ArgumentParser(prog="erimsim")
    p.add_argument("--exp", required=True, choices=["E1", "E2", "E3", "E4"])
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true", help="tiny sizes, CPU, under two minutes")
    p.add_argument("--mini", action="store_true", help="intermediate sizes for a CPU sanity run")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--mem-fraction", default=None, help="XLA_PYTHON_CLIENT_MEM_FRACTION (e.g. 0.45 for two jobs per GPU)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard", default=None, help="k/n: run units with index %% n == k")
    p.add_argument("--units", default=None, help="glob on unit ids")
    p.add_argument("--seeds", default=None, help="comma-separated seeds overriding the config")
    p.add_argument("--list", action="store_true", help="print the units and exit")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if a.mem_fraction:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(a.mem_fraction)
    import jax
    from . import io
    from .config import preset, smoke, mini
    from dataclasses import replace
    c = preset(a.exp)
    if a.smoke:
        c = smoke(c)
    elif a.mini:
        c = mini(c)
    if a.seeds:
        c = replace(c, exp=replace(c.exp, seeds=tuple(int(s) for s in a.seeds.split(","))))
    mod = {"E1": "e1_main", "E2": "e2_rates", "E3": "e3_dual", "E4": "e4_rl"}[a.exp]
    exp = __import__(f"erimsim.experiments.{mod}", fromlist=["units", "run_unit"])
    out = os.path.join(a.out, a.exp)
    os.makedirs(out, exist_ok=True)
    units = exp.units(c)
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        units = [u for i, u in enumerate(units) if i % n == k]
    if a.units:
        units = [u for u in units if fnmatch.fnmatch(u, a.units)]
    if a.list:
        print("\n".join(units))
        return 0
    logf = open(os.path.join(out, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    io.write_json(os.path.join(out, "meta.json"), {"exp": a.exp, "smoke": a.smoke, "config": c.to_json(), "git": io.git_hash(),
                                                   "device": io.device_info(), "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"{a.exp}: {len(units)} units on {io.device_info()}")
    failed = 0
    for u in units:
        if a.resume and io.is_done(out, u):
            log(f"skip {u} (done)")
            continue
        t0 = time.time()
        log(f"start {u}")
        try:
            info = exp.run_unit(c, u, out, log=log)
            io.mark_done(out, u)
            log(f"done  {u} in {time.time() - t0:.0f}s: {info}")
        except Exception:
            failed += 1
            log(f"FAILED {u}\n{traceback.format_exc()}")
    log(f"finished, {failed} failed")
    return 1 if failed else 0
