"""End-to-end smoke test of the runner on E1 (the other experiments are exercised by scripts/bench.sh)."""
import os
import tempfile
from erimsim.runner import main


def test_e1_smoke_runs_and_writes_outputs():
    with tempfile.TemporaryDirectory() as d:
        rc = main(["--exp", "E1", "--out", d, "--smoke"])
        assert rc == 0
        e1 = os.path.join(d, "E1")
        for f in ("results.csv", "training.csv", "trajectories_sample.csv", "meta.json", "stage_logs_baseline.npz"):
            assert os.path.exists(os.path.join(e1, f)), f
        lines = open(os.path.join(e1, "results.csv")).read().strip().splitlines()
        assert lines[0].startswith("method,seed,mission,cls") and len(lines) == 1 + 2 * 8      # baseline + erim, 8 missions
        assert os.path.exists(os.path.join(e1, "units", "eval_erim_s0.done"))
        # resume skips everything
        rc = main(["--exp", "E1", "--out", d, "--smoke", "--resume"])
        assert rc == 0 and len(open(os.path.join(e1, "results.csv")).read().strip().splitlines()) == 1 + 2 * 8
