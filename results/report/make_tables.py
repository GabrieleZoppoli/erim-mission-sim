#!/usr/bin/env python3
"""Write every table of the operator's report as LaTeX fragments from the CSVs and logs in results/.
usage: python3 make_tables.py <results_dir> <out_dir>
Nothing is typed by hand: the fragments are \\input by report.tex. Statistics are labelled mean or median in the headers;
the reconciliation against the log lines is printed to stdout for the operator to check."""
import sys, re, json, math, pathlib, datetime as dt
import pandas as pd, numpy as np

R = pathlib.Path(sys.argv[1]); O = pathlib.Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
def W(name, s): (O / name).write_text(s + ("%" if name.startswith(("num_", "par_")) else ""))
def f(x, d=3): return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"
def stopped_col(s): return s.astype(str).str.lower().isin(["1", "1.0", "true"]).astype(float)

# ---------------------------------------------------------------- E1 (run B in E1/, run A kept beside)
def e1_table(path, label):
    d = pd.read_csv(path); d["stopped_f"] = stopped_col(d["stopped"])
    rows = []
    for (m, s), g in d.groupby(["method", "seed"], sort=True):
        st = g[g.stopped_f == 1]
        rows.append(dict(method=m, seed=int(s), n=len(g), stopped=g.stopped_f.mean(), class_ok=g.class_ok_T.mean(),
                         fd_med=g.final_dist.median(), fd_mean=g.final_dist.mean(), cost_med=g.total_cost.median(),
                         cost_mean=g.total_cost.mean(), stop_t=st.stop_t.median() if len(st) else np.nan,
                         dist_stop=st.dist_stop.median() if len(st) and "dist_stop" in st else np.nan))
    t = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"method & seed & $n$ & stop rate & recog. at $T$ & dist. median (m) & dist. mean (m) & cost, mean & stop stage \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.method} & {'--' if r.seed < 0 else int(r.seed)} & {r.n} & {f(r.stopped)} & {f(r.class_ok)} & {f(r.fd_med)} & {f(r.fd_mean)} & {f(r.cost_mean,1)} & {f(r.stop_t,0)} \\\\")
    e = d[d.method == "erim"]; st = e[e.stopped_f == 1]; seeds = sorted(e.seed.unique())
    if len(seeds) > 1:
        lines.append(r"\midrule")
        lines.append(f"erim, pooled & {seeds[0]}--{seeds[-1]} & {len(e)} & {f(e.stopped_f.mean())} & {f(e.class_ok_T.mean())} & {f(e.final_dist.median())} & {f(e.final_dist.mean())} & {f(e.total_cost.mean(),1)} & {f(st.stop_t.median(),0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    W(f"tab_e1_{label}.tex", "\n".join(lines))
    print(f"[E1 {label}] baseline: stopped {t[t.method=='baseline'].stopped.iloc[0]:.3f} class_ok {t[t.method=='baseline'].class_ok.iloc[0]:.3f} fd_med {t[t.method=='baseline'].fd_med.iloc[0]:.4f} fd_mean {t[t.method=='baseline'].fd_mean.iloc[0]:.4f}")
    return t
e1b = e1_table(R / "E1/results.csv", "runB")
if (R / "E1_runA_cfa81fa/results.csv").exists(): e1a = e1_table(R / "E1_runA_cfa81fa/results.csv", "runA")
tr = pd.read_csv(R / "E1/training.csv")
lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule", r"seed & CNN loss & CNN accuracy & policy loss, best & policy loss, initial & improved & skipped steps \\", r"\midrule"]
for _, r in tr.iterrows(): lines.append(f"{int(r.seed)} & {f(r.cnn_loss)} & {f(r.cnn_acc)} & {f(r.pol)} & {f(r.pol_initial,1)} & {str(r.pol_improved)} & {int(r.pol_skipped)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e1_training.tex", "\n".join(lines))

# ---------------------------------------------------------------- E2
est = pd.read_csv(R / "E2/rates_est.csv")
g = est.groupby(["n", "M"]).agg(cost=("cost_test", "median"), fd=("final_dist", "median"), gap=("gap", "median"), imp=("pol_improved", lambda s: s.astype(str).isin(["True","1","1.0"]).mean()), sk=("pol_skipped", "sum")).reset_index()
ns = sorted(g.n.unique()); Ms = sorted(g.M.unique())
lines = [r"\begin{tabular}{r" + "r" * len(Ms) + "}", r"\toprule", "width $n$ / missions $M$ & " + " & ".join(str(m) for m in Ms) + r" \\", r"\midrule"]
for n in ns: lines.append(f"{n} & " + " & ".join(f(g[(g.n==n)&(g.M==m)].cost.iloc[0], 1) if len(g[(g.n==n)&(g.M==m)]) else "--" for m in Ms) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e2_cost.tex", "\n".join(lines))
lines = [r"\begin{tabular}{r" + "r" * len(Ms) + "}", r"\toprule", "width $n$ / missions $M$ & " + " & ".join(str(m) for m in Ms) + r" \\", r"\midrule"]
for n in ns: lines.append(f"{n} & " + " & ".join(f(g[(g.n==n)&(g.M==m)].fd.iloc[0], 3) if len(g[(g.n==n)&(g.M==m)]) else "--" for m in Ms) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e2_fd.tex", "\n".join(lines))
cnn = pd.read_csv(R / "E2/rates_cnn.csv"); gc = cnn.groupby(["J", "M"]).acc.median().reset_index()
Js = sorted(gc.J.unique()); Mc = sorted(gc.M.unique())
lines = [r"\begin{tabular}{r" + "r" * len(Mc) + "}", r"\toprule", "depth $J$ / missions $M$ & " + " & ".join(str(m) for m in Mc) + r" \\", r"\midrule"]
for j in Js: lines.append(f"{j} & " + " & ".join(f(gc[(gc.J==j)&(gc.M==m)].acc.iloc[0], 3) if len(gc[(gc.J==j)&(gc.M==m)]) else "--" for m in Mc) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e2_cnn.tex", "\n".join(lines))
ref_s = "The reference row is pending."
if (R / "E2/rates_ref.csv").exists():
    rf = pd.read_csv(R / "E2/rates_ref.csv").iloc[0]
    ref_s = (f"The high-capacity reference (width {int(rf.n)}, {int(rf.M)} training missions, {int(rf.n_params)} estimator parameters, CNN depth {int(rf.cnn_J)} with {int(rf.cnn_p_J)} parameters) "
             f"reaches a held-out cost of {rf.cost_test:.1f} against a training cost of {rf.cost_train:.1f}, a median final distance of {rf.final_dist:.3f} m, a policy loss of {rf.pol_loss:.3f} "
             f"after {int(rf.pol_skipped)} skipped steps, and a single-image CNN accuracy of {rf.cnn_acc:.3f} (near {rf.cnn_acc_near:.3f}, mid {rf.cnn_acc_mid:.3f}, far {rf.cnn_acc_far:.3f}).")
W("par_e2_ref.tex", ref_s)
W("num_e2.tex", f"{len(est)} estimator rows ({est.pol_improved.astype(str).isin(['True','1','1.0']).sum()} with an improved policy, {int(est.pol_skipped.sum())} skipped steps in all) and {len(cnn)} CNN rows")

# ---------------------------------------------------------------- E3 (+ control cost split)
e3 = pd.read_csv(R / "E3/results.csv"); cc = pd.read_csv(R / "E3/results_control_cost.csv")
e3["lam"] = e3.method.str.replace("dual_l", "").astype(float); cc["lam"] = cc.method.str.replace("dual_l", "").str.replace("_c0", "").astype(float)
rows = []
for (lam, s), gg in e3.groupby(["lam", "seed"], sort=True):
    c = cc[(cc.lam == lam) & (cc.seed == s)]
    tf = gg.t_first_correct; never = int((tf < 0).sum()); tf_med = tf[tf >= 0].median()
    rows.append(dict(lam=lam, seed=int(s), n=len(gg), total=gg.total_cost.mean(), control=c.total_cost.mean() if len(c) else np.nan,
                     ent=(c.entropy_sum.mean() if len(c) and "entropy_sum" in c else np.nan), fd=gg.final_dist.median(), cls=gg.class_ok_T.mean(), tfc=tf_med, never=never))
t3 = pd.DataFrame(rows)
lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule", r"$\lambda$ & seed & total cost & control cost & entropy sum & info. term & dist. median (m) & recog. at $T$ & first-correct stage (never) \\", r"\midrule"]
for _, r in t3.iterrows():
    lines.append(f"{r.lam:g} & {int(r.seed)} & {f(r.total,2)} & {f(r.control,2)} & {f(r.ent,2)} & {f(max(r.total - r.control, 0.0),2)} & {f(r.fd)} & {f(r.cls)} & {f(r.tfc,0)} ({int(r.never)}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e3.tex", "\n".join(lines))
print("[E3] total mean per (lam,seed):", {(r.lam, r.seed): round(r.total, 2) for _, r in t3.iterrows()})

# ---------------------------------------------------------------- E4 (two seeds; seed 0 best iterate only in the rerun directory)
def e4_rows(path, tag):
    d = pd.read_csv(path); out = []
    for (m, s), gg in d.groupby(["method", "seed"], sort=True):
        out.append(dict(run=tag, method=m, seed=int(s), n=len(gg), fd_med=gg.final_dist.median(), fd_mean=gg.final_dist.mean(), cost=gg.total_cost.mean(), cls=gg.class_ok_T.mean() if "class_ok_T" in gg else np.nan))
    return out
r4 = e4_rows(R / "E4/results.csv", "main")
if (R / "E4_s0_cafa903/E4/results.csv").exists(): r4 += e4_rows(R / "E4_s0_cafa903/E4/results.csv", "seed-0 rerun")
t4 = pd.DataFrame(r4)
pb = pd.read_csv(R / "E4/ppo_best.csv") if (R / "E4/ppo_best.csv").exists() else pd.DataFrame()
pb2 = pd.read_csv(R / "E4_s0_cafa903/E4/ppo_best.csv") if (R / "E4_s0_cafa903/E4/ppo_best.csv").exists() else pd.DataFrame()
best = {("main", int(r.seed)): int(r.best_update) for _, r in pb.iterrows()}; best.update({("seed-0 rerun", int(r.seed)): int(r.best_update) for _, r in pb2.iterrows()})
lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"run & policy & seed & $n$ & final distance, median (m) & final distance, mean (m) & cost, mean \\", r"\midrule"]
for _, r in t4.iterrows():
    lab = {"ppo": "final iterate", "ppo_best": f"best iterate (update {best.get((r.run, r.seed), '?')})"}.get(r.method, r.method)
    lines.append(f"{r.run} & {lab} & {r.seed} & {r.n} & {f(r.fd_med)} & {f(r.fd_mean,1)} & {f(r.cost,0)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e4.tex", "\n".join(lines))

# ---------------------------------------------------------------- hard regime (optional)
H = R / "hard"
if (H / "E1/results.csv").exists():
    e1h = e1_table(H / "E1/results.csv", "hard")
if (H / "E3/results.csv").exists():
    e3h = pd.read_csv(H / "E3/results.csv"); e3h["lam"] = e3h.method.str.replace("dual_l", "").astype(float)
    cch = pd.read_csv(H / "E3/results_control_cost.csv") if (H / "E3/results_control_cost.csv").exists() else pd.DataFrame(columns=["method","seed","total_cost","entropy_sum"])
    if len(cch): cch["lam"] = cch.method.str.replace("dual_l", "").str.replace("_c0", "").astype(float)
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule", r"$\lambda$ & seed & $n$ & total cost & control cost & entropy sum & dist. median (m) & recog. at $T$ & first-correct stage (never) \\", r"\midrule"]
    for (lam, s), gg in e3h.groupby(["lam", "seed"], sort=True):
        c = cch[(cch.lam == lam) & (cch.seed == s)] if len(cch) else cch
        tf = gg.t_first_correct; never = int((tf < 0).sum())
        lines.append(f"{lam:g} & {int(s)} & {len(gg)} & {f(gg.total_cost.mean(),2)} & {f(c.total_cost.mean() if len(c) else np.nan,2)} & {f(c.entropy_sum.mean() if len(c) else np.nan,2)} & {f(gg.final_dist.median())} & {f(gg.class_ok_T.mean())} & {f(tf[tf>=0].median(),0)} ({never}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; W("tab_e3_hard.tex", "\n".join(lines))
    W("flag_hard.tex", r"\def\hardflag{1}")
else:
    W("tab_e3_hard.tex", r"\emph{Hard-regime results pending.}"); W("flag_hard.tex", r"\def\hardflag{0}")
    if not (H / "E1/results.csv").exists(): W("tab_e1_hard.tex", r"\emph{Hard-regime results pending.}")

# ---------------------------------------------------------------- GPU-hour accounting: device occupancy = wall-clock span of each stream's log
# (first to last [HH:MM:SS] stamp, midnight-safe); streams on one GPU ran one after the other, except the two E2 estimator shards,
# which ran concurrently and are measured as the span of their union. Unit time ("done X in Ns") is given beside it for reference.
TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]", re.M)
def stamps(logs):
    secs = []
    for log in logs:
        p = R / log
        if not p.exists(): continue
        prev, day_off = None, 0
        for m in TS.finditer(p.read_text(errors="ignore")):
            s = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + day_off
            if prev is not None and s < prev - 3600: day_off += 86400; s += 86400   # crossed midnight within the log
            secs.append(s); prev = s
    return secs
def span_h(logs):
    s = stamps(logs); return (max(s) - min(s)) / 3600 if len(s) >= 2 else 0.0
def units_h(logs):
    tot, k = 0.0, 0
    for log in logs:
        p = R / log
        if p.exists():
            for m in re.finditer(r"done +\S+ in (\d+)s", p.read_text(errors="ignore")): tot += int(m.group(1)); k += 1
    return tot / 3600, k
streams = [("GPU 0", "bench: environment report, smoke tests, benchmark (b727351)", ["bench_run.log"], 0.0),
           ("GPU 0", "first run: E1 (void) and E2 CNN sweep (b727351)", ["gpu0.log"], 0.0),
           ("GPU 0", "E1 run A (cfa81fa)", ["gpu0_rerun.log"], 0.0),
           ("GPU 0", "E3 partial, void (1b306c9)", ["gpu0_rerun2_void_1b306c9.log"], 0.0),
           ("GPU 0", "E1 re-evaluation, E3, ref attempt (7161ce6)", ["gpu0_rerun3.log"], 0.0),
           ("GPU 0", "E1 run B (9af9f53)", ["gpu0_e1_runB.log"], 0.0),
           ("GPU 0", "E3 re-evaluation (8ff851f)", ["e3_reeval.log"], 0.0),
           ("GPU 0", "E2 reference (6b3b395)", ["e2_ref.log"], 0.0),
           ("GPU 1", "first run: E2 estimator sweep, void, two shards (b727351)", ["e2_shard0_void_firstrun.log", "e2_shard1_void_firstrun.log"], 0.0),
           ("GPU 1", "first run: E4 seed 0 (void) and 14 min of seed 1 (b727351)", ["gpu1.log"], 0.0),
           ("GPU 1", "E2 estimator sweep, two shards (9af9f53)", ["e2_shard0.log", "e2_shard1.log"], 0.0),
           ("GPU 1", "E4 seed 0 (9af9f53)", ["gpu1_e4_rerun.log"], 0.0),
           ("GPU 1", "E4 seed 1 (cafa903)", ["e4_s1.log"], 0.0),
           ("GPU 1", "E4 seed 0 rerun (cafa903)", ["e4_s0_rerun.log"], 0.0),
           ("GPU 1", "hard regime: E1 seed 0 (9dcef48)", ["hard/e1_hard.log"], 0.0),
           ("GPU 1", "hard regime: E3 lambda 0 and 1, seed 0 (9dcef48)", ["hard/e3_hard.log"], 0.0),
           ("GPU 1", "hard regime: re-evaluation (9dcef48)", ["hard/e3_reeval_hard.log"], 0.0)]
tot = {"GPU 0": 0.0, "GPU 1": 0.0}; lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"stream (commit) & units & unit time (h) & occupancy (h) \\", r"\midrule"]
cur = None
for gpu, lab, logs, extra in streams:
    if not any((R / l).exists() for l in logs): continue
    if gpu != cur: lines.append(r"\multicolumn{4}{l}{\emph{" + gpu + r"}} \\"); cur = gpu
    uh, k = units_h(logs); oh = span_h(logs) + extra; tot[gpu] += oh
    lines.append(f"{lab} & {k} & {uh:.2f} & {oh:.2f} \\\\")
lines += [r"\midrule", f"GPU 0 total & & & {tot['GPU 0']:.1f} \\\\", f"GPU 1 total & & & {tot['GPU 1']:.1f} \\\\", f"both GPUs & & & {tot['GPU 0'] + tot['GPU 1']:.1f} \\\\", r"\bottomrule", r"\end{tabular}"]
W("tab_gpu_hours.tex", "\n".join(lines))
print(f"[GPU-hours] occupancy GPU0 {tot['GPU 0']:.2f} h, GPU1 {tot['GPU 1']:.2f} h, total {tot['GPU 0'] + tot['GPU 1']:.2f} h")
json.dump(dict(gpu0_h=tot["GPU 0"], gpu1_h=tot["GPU 1"], total_h=tot["GPU 0"] + tot["GPU 1"]), open(O / "gpu_hours.json", "w"), indent=1)
W("num_gpu_total.tex", f"{tot['GPU 0'] + tot['GPU 1']:.1f}"); W("num_gpu0.tex", f"{tot['GPU 0']:.1f}"); W("num_gpu1.tex", f"{tot['GPU 1']:.1f}")
print("done; fragments in", O)
