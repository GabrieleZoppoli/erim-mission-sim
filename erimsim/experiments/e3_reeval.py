"""E3 post-processing: re-evaluate every saved dual policy (out/E3/weights/dual_l<lambda>_s<seed>.npz) with lambda = 0
on the same evaluation missions, so that the control cost (squared distance plus control effort) is separated from the
information term lambda * sum_t entropy(posterior_t), which the E3 results.csv rows include in total_cost.

Rows go to out/E3/results_control_cost.csv with method "dual_l<lambda>_c0", the columns of results.csv plus `lambda`,
`entropy_sum` (sum over the T stages of the posterior entropy), `entropy_mean` and `entropy_T`. For every unit the
identity total_cost(results.csv) = total_cost(control, this file) + lambda * entropy_sum holds up to float rounding,
since the rollout is deterministic given the policy and the mission keys. One done marker per unit; rerunnable.

    python3 -m erimsim.experiments.e3_reeval --out DIR [--gpu N] [--smoke]       (about 40 s per unit on an A6000)
"""
import argparse
import glob
import os
import re
import sys
import time


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--smoke", action="store_true", help="the smoke configuration (for a run produced with --smoke)")
    p.add_argument("--mini", action="store_true")
    p.add_argument("--hard", action="store_true", help="hard-recognition regime (config.hard), for a run produced with --hard")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import numpy as np
    import jax.numpy as jnp
    from .. import io
    from ..config import preset, smoke, mini, hard
    from ..erim import agent as eagent
    from . import common, e1_main

    c = preset("E3")
    if a.smoke:
        c = smoke(c)
    elif a.mini:
        c = mini(c)
    if a.hard:
        c = hard(c)
    e3 = os.path.join(a.out, "E3")
    e1 = os.path.join(a.out, "E1")
    files = sorted(glob.glob(os.path.join(e3, "weights", "dual_l*_s*.npz")))
    if not files:
        print(f"no dual weights under {os.path.join(e3, 'weights')}", file=sys.stderr)
        return 1
    logf = open(os.path.join(e3, f"log_reeval_{time.strftime('%Y%m%d_%H%M%S')}.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    missions, h = common.eval_missions(c)
    log(f"E3 re-evaluation with lambda = 0: {len(files)} policies, missions {h}, {io.device_info()}")
    A = eagent.bind(c.sim, c.model, use_images=True, use_post=True)
    like = eagent.init_params(__import__("jax").random.PRNGKey(0), c.sim, c.model, e1_main._zero_norm(c))
    failed = 0
    for f in files:
        m = re.match(r"dual_l(?P<lam>[0-9.eE+-]+)_s(?P<seed>\d+)\.npz$", os.path.basename(f))
        if not m:
            log(f"skip {f}: unexpected name")
            continue
        lam, seed = float(m["lam"]), int(m["seed"])
        unit = f"reeval_dual_l{lam:g}_s{seed}"
        if io.is_done(e3, unit):
            log(f"skip {unit} (done)")
            continue
        t0 = time.time()
        try:
            params = io.load_pytree(f, like)
            rows, logs = common.evaluate(A, params, c, missions, f"dual_l{lam:g}_c0", seed, chunk=200, lam=0.0)
            ent = np.asarray(logs.entropy, dtype=np.float64)                 # (missions, T)
            for r, e in zip(rows, ent):
                r["lambda"] = lam
                r["entropy_sum"] = float(e.sum()); r["entropy_mean"] = float(e.mean()); r["entropy_T"] = float(e[-1])
            io.append_csv(os.path.join(e3, "results_control_cost.csv"), rows)
            io.mark_done(e3, unit)
            cc = float(np.mean([r["total_cost"] for r in rows])); es = float(np.mean([r["entropy_sum"] for r in rows]))
            log(f"done  {unit} in {time.time() - t0:.0f}s: lambda {lam:g}, seed {seed}, control cost {cc:.2f}, "
                f"entropy_sum {es:.2f}, implied total {cc + lam * es:.2f}, final_dist {np.median([r['final_dist'] for r in rows]):.4f}")
        except Exception as ex:                                        # keep going; the chain must not stop on one unit
            failed += 1
            log(f"FAILED {unit}: {ex!r}")
    log(f"finished, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
