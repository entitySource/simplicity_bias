"""Summarise runs.jsonl into tables (results/summary.md, results/*.csv) and figures.

Usage: python analyze.py [path/to/runs.jsonl]
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
RNG = np.random.default_rng(0)
COL = {2: "#7f7f7f", 3: "#1f77b4", 5: "#d62728"}


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cond_key(r):
    d, t = r["data"], r["train"]
    return (r["exp"], t.get("opt", "sgd"), t.get("depth", 1), d.get("m", 3),
            d.get("alpha_s", 1.0), d.get("alpha_c", 1.0), d.get("d_noise", 20))


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    return 2.776 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) == 5 else 1.96 * x.std(ddof=1) / math.sqrt(len(x))


def speed_table(runs):
    """One row per condition: isolated learning times, speed gap, joint preference."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        if r["exp"] in ("E1", "E3", "R1"):
            g[cond_key(r)][r["mode"]].append(r)
    rows = []
    for k, modes in sorted(g.items()):
        if not all(m in modes for m in ("only_s", "only_c", "joint")):
            continue
        ts = [r["t95"] for r in modes["only_s"]]
        tc = [r["t95"] for r in modes["only_c"]]
        censored = any(t is None for t in ts + tc)
        Ts = float(np.median([t if t is not None else np.inf for t in ts]))
        Tc = float(np.median([t if t is not None else np.inf for t in tc]))
        Ts, Tc = max(Ts, 1.0), max(Tc, 1.0)
        ps = [r["p_simple"] for r in modes["joint"]]
        rows.append(dict(
            exp=k[0], opt=k[1], depth=k[2], m=k[3], alpha_s=k[4], alpha_c=k[5], d_noise=k[6],
            T_s=Ts, T_c=Tc, gap=math.log2(Tc / Ts) if np.isfinite(Tc / Ts) else float("nan"),
            censored=censored, n=len(ps), ts_seeds=ts, tc_seeds=tc, P_s=float(np.mean(ps)), P_s_ci=ci95(ps), P_s_seeds=ps,
            iso_s_test=float(np.mean([r["test_acc"] for r in modes["only_s"]])),
            iso_c_test=float(np.mean([r["test_acc"] for r in modes["only_c"]])),
            joint_test=float(np.mean([r["test_acc"] for r in modes["joint"]])),
            drop_s=float(np.mean([r["test_acc"] - r["acc_shuffle_s"] for r in modes["joint"]])),
            drop_c=float(np.mean([r["test_acc"] - r["acc_shuffle_c"] for r in modes["joint"]])),
        ))
    return rows


def valid(r):
    """A cell enters the fit only if both isolated runs reached 95% train accuracy
    within budget AND generalised (isolated test accuracy >= 0.95). Otherwise the
    'speed' of that feature is the speed of memorising noise, not of learning it."""
    return (not r["censored"] and np.isfinite(r["gap"])
            and r["iso_s_test"] >= 0.95 and r["iso_c_test"] >= 0.95)


G_GRID = np.linspace(-5, 5, 1001)
B_GRID = np.exp(np.linspace(np.log(0.2), np.log(20), 60))


def fit_sigmoid(x, y):
    """Least squares fit of y = sigmoid(b * (x - g)) by grid search (robust to
    saturated points, unlike OLS on logits). Returns (g, b)."""
    z = B_GRID[:, None, None] * (x[None, None, :] - G_GRID[None, :, None])
    pred = 1 / (1 + np.exp(-z))
    sse = ((pred - y[None, None, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    return G_GRID[j], B_GRID[i]


def indifference(rows, n_boot=500):
    """Fit P_s = sigmoid(b * (gap - gap*)) per m on seed-level points.

    gap* is the isolated-speed gap at which the network is indifferent between
    the two features. gap* = 0 means speed alone predicts the preference;
    gap* < 0 means the complex feature must be learned *faster* than the simple
    one before it is preferred (a residual simplicity preference).
    CIs come from resampling seeds within each condition.
    """
    rows = [r for r in rows if valid(r)]
    if len(rows) < 3:
        return None
    gaps = np.array([r["gap"] for r in rows])
    seeds = [np.array(r["P_s_seeds"]) for r in rows]

    ts = [np.array(r["ts_seeds"], float) for r in rows]
    tc = [np.array(r["tc_seeds"], float) for r in rows]

    def fit(ps_list, gap_list):
        x = np.concatenate([np.full(len(p), g) for g, p in zip(gap_list, ps_list)])
        return fit_sigmoid(x, np.concatenate(ps_list))

    def rgap(a, b):  # bootstrap the speed gap too: resample isolated-run seeds
        return math.log2(max(np.median(RNG.choice(b, len(b))), 1) / max(np.median(RNG.choice(a, len(a))), 1))

    est, slope = fit(seeds, gaps)
    boots = np.array([fit([RNG.choice(p, size=len(p)) for p in seeds],
                          [rgap(a, b) for a, b in zip(ts, tc)])[0] for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # is there data on both sides of P = 0.5? otherwise gap* is an extrapolation
    ps = np.array([r["P_s"] for r in rows])
    bracketed = bool((ps > 0.5).any() and (ps < 0.5).any())
    return dict(gap_star=est, lo=lo, hi=hi, slope=slope, n_cond=len(rows), bracketed=bracketed,
                gap_range=(float(gaps.min()), float(gaps.max())))


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main(path):
    runs = load(path)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = speed_table(runs)
    out = []

    # ---------- E1 / E3 / R1 tables + indifference points ----------
    with open(os.path.join(RES, "speed_vs_preference.csv"), "w") as f:
        cols = ["exp", "opt", "depth", "m", "alpha_s", "alpha_c", "d_noise", "T_s", "T_c", "gap",
                "censored", "n", "P_s", "P_s_ci", "iso_s_test", "iso_c_test", "joint_test", "drop_s", "drop_c"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    out.append("# Results summary (generated by analyze.py)\n")
    out.append("P_s = fraction of conflict test points classified by the simple feature "
               "(1 = always simple, 0 = always complex). gap = log2(T_c / T_s), where T is the "
               "median number of SGD steps to 95% train accuracy when only that feature is "
               "predictive. gap > 0 means the complex feature is slower to learn in isolation.\n")
    out.append("## Indifference points (gap* where P_s = 0.5)\n")
    out.append("Fit: P_s = sigmoid(b * (gap - gap*)) by least squares on seed-level values, "
               "using only valid cells (both isolated runs reached 95% train accuracy and >= 95% "
               "test accuracy). CI: 500 bootstrap resamples of seeds within cells, resampling both the joint runs (P_s) and the isolated runs (gap). "
               "'bracketed' = valid cells exist on both sides of P_s = 0.5.\n")
    out.append("| exp | optimizer | depth | m | gap* | 95% CI | slope b | valid cells | bracketed | gap range |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    ind = {}
    for exp in ("E1", "E3", "R1"):
        for opt in ("sgd", "adam"):
            for depth in (1, 3):
                for m in (2, 3, 5):
                    sub = [r for r in rows if (r["exp"], r["opt"], r["depth"], r["m"]) == (exp, opt, depth, m)]
                    if not sub:
                        continue
                    res = indifference(sub)
                    if res is None:
                        continue
                    ind[(exp, opt, depth, m)] = res
                    out.append(f"| {exp} | {opt} | {depth} | {m} | {fmt(res['gap_star'])} | "
                               f"[{fmt(res['lo'])}, {fmt(res['hi'])}] | {fmt(res['slope'])} | {res['n_cond']} | {res['bracketed']} | "
                               f"{fmt(res['gap_range'][0])} to {fmt(res['gap_range'][1])} |")
    out.append("")

    for exp in ("E1", "E3", "R1"):
        sub = [r for r in rows if r["exp"] == exp]
        if not sub:
            continue
        out.append(f"## {exp}: per-condition table\n")
        out.append("| opt | depth | m | a_s | a_c | T_s | T_c | gap | P_s (±95% CI) | iso_s test | iso_c test | joint test |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            cen = " (censored)" if r["censored"] else ("" if valid(r) else " (excluded)")
            out.append(f"| {r['opt']} | {r['depth']} | {r['m']} | {r['alpha_s']} | {r['alpha_c']} | "
                       f"{r['T_s']:.0f} | {r['T_c']:.0f}{cen} | {fmt(r['gap'])} | "
                       f"{r['P_s']:.3f} ± {fmt(r['P_s_ci'], 3)} | {r['iso_s_test']:.3f} | "
                       f"{r['iso_c_test']:.3f} | {r['joint_test']:.3f} |")
        out.append("")

    # Figure 1: E1 preference vs speed gap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, opt in zip(axes, ("sgd", "adam")):
        for m in (2, 3, 5):
            allsub = [r for r in rows if (r["exp"], r["opt"], r["depth"], r["m"]) == ("E1", opt, 1, m)
                      and np.isfinite(r["gap"])]
            sub = [r for r in allsub if valid(r)]
            bad = [r for r in allsub if not valid(r)]
            ax.plot([r["gap"] for r in bad], [r["P_s"] for r in bad], "x", color=COL[m], alpha=0.5, ms=5)
            if not sub:
                continue
            ax.errorbar([r["gap"] for r in sub], [r["P_s"] for r in sub],
                        yerr=[r["P_s_ci"] for r in sub], fmt="o", color=COL[m], alpha=0.8,
                        ms=5, capsize=2, label={2: "m=2 (both linear, control)", 3: "m=3 slabs", 5: "m=5 slabs"}[m])
            # natural-scale point (alpha_s = alpha_c = 1) marked with a ring
            nat = [r for r in sub if r["alpha_s"] == 1 and r["alpha_c"] == 1]
            for r in nat:
                ax.plot(r["gap"], r["P_s"], "o", ms=12, mfc="none", mec=COL[m], mew=1.5)
            res = ind.get(("E1", opt, 1, m))
            if res and sub:
                xs = np.linspace(min(r["gap"] for r in sub), max(r["gap"] for r in sub), 100)
                ys = 1 / (1 + np.exp(-res["slope"] * (xs - res["gap_star"])))
                ax.plot(xs, ys, "-", color=COL[m], lw=1, alpha=0.6)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"E1, 1 hidden layer, {opt.upper()}")
        ax.set_xlim(-6, 6.5)
        ax.set_xlabel("isolated speed gap  log2(T_complex / T_simple)\n<- complex learned faster      complex learned slower ->")
    axes[0].set_ylabel("P_s: share of conflict points decided by simple feature")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.text(0.5, -0.04, "Rings: natural scale (alpha_s = alpha_c = 1). x: excluded cells (a feature not learned in isolation). "
             "Lines: per-m sigmoid fit. "
             "Error bars: 95% CI over 5 seeds.", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_preference_vs_speed.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- E2: predictive strength ----------
    e2 = [r for r in runs if r["exp"] == "E2"]
    if e2:
        g = collections.defaultdict(list)
        for r in e2:
            g[(r["train"]["opt"], r["data"]["m"], r["data"]["p_s"])].append(r)
        out.append("## E2: lowering the simple feature's predictive strength (p_c = 1, natural scale, 8000 steps)\n")
        out.append("| opt | m | p_s | P_s final (±95% CI) | P_s at step 500 | train acc | clean test acc | acc drop if simple shuffled | acc drop if complex shuffled |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for (opt, m, p), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])):
            ps = [r["p_simple"] for r in rs]
            i500 = rs[0]["curve"]["step"].index(500)
            out.append(f"| {opt} | {m} | {p} | {np.mean(ps):.3f} ± {fmt(ci95(ps), 3)} | "
                       f"{np.mean([r['curve']['p_simple'][i500] for r in rs]):.3f} | "
                       f"{np.mean([r['final_train_acc'] for r in rs]):.3f} | {np.mean([r['test_acc'] for r in rs]):.3f} | "
                       f"{np.mean([r['test_acc'] - r['acc_shuffle_s'] for r in rs]):.3f} | "
                       f"{np.mean([r['test_acc'] - r['acc_shuffle_c'] for r in rs]):.3f} |")
        out.append("")
        for ax, opt in zip(axes, ("sgd", "adam")):
            for m in (3, 5):
                pts = sorted((p, [r["p_simple"] for r in rs]) for (o, mm, p), rs in g.items() if o == opt and mm == m)
                if not pts:
                    continue
                ax.errorbar([p for p, _ in pts], [np.mean(v) for _, v in pts], yerr=[ci95(v) for _, v in pts],
                            fmt="o-", color=COL[m], capsize=2, label=f"m={m} slabs")
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.invert_xaxis()
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("p_s: agreement of simple feature with label (complex feature: 1.0)")
            ax.set_title(f"E2, {opt.upper()}, 8000 steps")
        axes[0].set_ylabel("P_s (final)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig2_predictive_strength.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Figure 3: dynamics of P_s for E2, SGD, m=3
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, m in zip(axes, (3, 5)):
            for p, c in zip([1.0, 0.95, 0.9, 0.8, 0.7, 0.6], plt.cm.viridis(np.linspace(0, 0.9, 6))):
                rs = g.get(("sgd", m, p))
                if not rs:
                    continue
                st = rs[0]["curve"]["step"]
                ax.plot(st, np.mean([r["curve"]["p_simple"] for r in rs], 0), color=c, label=f"p_s={p}")
            ax.set_xscale("symlog", linthresh=100)
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.set_title(f"P_s during training, SGD, m={m}")
            ax.set_xlabel("step")
        axes[0].set_ylabel("P_s")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig3_dynamics_predictive_strength.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "runs.jsonl"))
