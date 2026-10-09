"""E3 re-evaluation with lambda = 0 (post-processing): the control cost it reports plus lambda times the entropy sum
must reproduce the total cost of the E3 run itself, unit by unit, since the rollout is deterministic."""
import csv
import os
import tempfile
from erimsim.runner import main as run
from erimsim.experiments.e3_reeval import main as reeval


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_control_cost_plus_information_term_matches_total_cost():
    with tempfile.TemporaryDirectory() as d:
        assert run(["--exp", "E3", "--out", d, "--smoke"]) == 0
        assert reeval(["--out", d, "--smoke"]) == 0
        e3 = os.path.join(d, "E3")
        R = _rows(os.path.join(e3, "results.csv"))
        C = _rows(os.path.join(e3, "results_control_cost.csv"))
        assert len(C) == len(R) > 0
        key = lambda r: (r["method"].replace("_c0", ""), r["seed"], r["mission"])
        Rk = {key(r): r for r in R}
        for c in C:
            r = Rk[key(c)]
            lam = float(c["lambda"])
            implied = float(c["total_cost"]) + lam * float(c["entropy_sum"])
            assert abs(implied - float(r["total_cost"])) <= 1e-3 * max(1.0, abs(float(r["total_cost"]))), (c["method"], c["mission"], implied, r["total_cost"])
            assert c["final_dist"] == r["final_dist"] or abs(float(c["final_dist"]) - float(r["final_dist"])) < 1e-4
        # rerun skips everything
        assert reeval(["--out", d, "--smoke"]) == 0
        assert len(_rows(os.path.join(e3, "results_control_cost.csv"))) == len(C)
