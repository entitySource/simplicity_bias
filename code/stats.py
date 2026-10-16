"""Inferential statistics on top of analyze.py (added after peer review of the pilot).

What it adds to the per-m indifference points in analyze.py:
  1. A direct test of the quantity the hypotheses are about, the *offset*
     delta(m) = gap*(m) - gap*(2), with a bootstrap CI and p-value, instead of
     asking whether two separate CIs overlap (which is conservative).
  2. Holm correction across the (optimiser x m) family within each depth.
  3. An equivalence check against a smallest effect size of interest (SESOI):
     the offset is called "negligible" only if its 90% CI lies inside +-SESOI.
  4. A pooled offset per depth (mean over optimisers and m = 3, 5): one primary
     test per depth instead of four.
  5. A slope check: H_speed says the speed gap is a sufficient statistic, so the
     sigmoid slope should not depend on m.
  6. Sensitivity of everything to the validity rule (which was added post hoc).
  7. A sample-size projection from the pilot's bootstrap standard errors.

Usage: python stats.py [runs.jsonl ...] [--exps E1,E3,R1] [--out ../results/stats.md]
"""
from __future__ import annotations

import argparse
import math
import os
import zlib

import numpy as np

import analyze as A

SESOI = 0.5      # log2 units: a 1.41x difference in isolated learning speed
N_BOOT = 2000
Z = {0.95: 1.959964, 0.90: 1.644854, 0.80: 0.841621}


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - k) * pvals[i]))
        adj[i] = running
    return adj


def boot_p(d):
    """Two-sided bootstrap p-value for H0: offset = 0 (percentile method)."""
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d)))


def fits(rows, exps, rule):
    old = A.valid
    if rule == "censoring_only":
        A.valid = lambda r: not r["censored"] and np.isfinite(r["gap"])
    out = {}
    try:
        for exp in exps:
            for opt in ("sgd", "adam"):
                for depth in (1, 3):
                    for m in (2, 3, 5):
                        sub = [r for r in rows if (r["exp"], r["opt"], r["depth"], r["m"]) == (exp, opt, depth, m)]
                        if not sub:
                            continue
                        seed = zlib.crc32(repr((exp, opt, depth, m)).encode())
                        res = A.indifference(sub, n_boot=N_BOOT, seed=seed)
                        if res is not None:
                            out[(exp, opt, depth, m)] = res
    finally:
        A.valid = old
    return out


def offsets(ind, exps):
    """delta(m) = gap*(m) - gap*(2) per (exp, opt, depth, m), plus pooled per (exp, depth)."""
    rows = []
    for exp in exps:
        for depth in (1, 3):
            fam = []
            for opt in ("sgd", "adam"):
                c = ind.get((exp, opt, depth, 2))
                for m in (3, 5):
                    r = ind.get((exp, opt, depth, m))
                    if c is None or r is None:
                        continue
                    d = r["boots"] - c["boots"]          # independent data -> difference of draws
                    ls = np.log(r["bslopes"]) - np.log(c["bslopes"])
                    fam.append(dict(exp=exp, depth=depth, opt=opt, m=str(m),
                                    est=r["gap_star"] - c["gap_star"], draws=d,
                                    slope_ratio=r["slope"] / c["slope"], ls_draws=ls))
            if not fam:
                continue
            for f, a in zip(fam, holm(np.array([boot_p(f["draws"]) for f in fam]))):
                f["p_holm"] = a
            pooled = np.mean([f["draws"] for f in fam], 0)
            fam.append(dict(exp=exp, depth=depth, opt="pooled", m="3,5",
                            est=float(np.mean([f["est"] for f in fam])), draws=pooled,
                            slope_ratio=float("nan"), ls_draws=None, p_holm=float("nan")))
            rows += fam
    for f in rows:
        d = f["draws"]
        f["se"] = float(d.std(ddof=1))
        f["ci95"] = np.percentile(d, [2.5, 97.5])
        f["ci90"] = np.percentile(d, [5, 95])
        f["p"] = boot_p(d)
        f["equiv"] = bool(-SESOI < f["ci90"][0] and f["ci90"][1] < SESOI)
        if f["ls_draws"] is not None:
            f["slope_ci"] = np.exp(np.percentile(f["ls_draws"], [2.5, 97.5]))
    return rows


def verdict(f):
    lo, hi = f["ci95"]
    if hi < 0:
        v = "simplicity offset (CI < 0)"
    elif lo > 0:
        v = "anti-simplicity offset (CI > 0)"
    else:
        v = "not distinguishable from 0"
    return v + (f"; negligible (90% CI within +-{SESOI})" if f["equiv"] else "")


def power_rows(offs, n_now):
    """Seeds per cell needed for 80% power (two-sided alpha = .05), assuming the
    bootstrap SE shrinks as 1/sqrt(seeds) with the grid held fixed (a normal
    approximation; a denser grid near the crossover shrinks it further)."""
    out = []
    for f in offs:
        se = f["se"]
        mde = (Z[0.95] + Z[0.80]) * se
        need = {d: math.ceil(n_now * (se * (Z[0.95] + Z[0.80]) / d) ** 2) for d in (0.3, 0.5)}
        out.append((f, mde, need))
    return out


def fmt_ci(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def retime(runs, metric):
    """Replace T95 by the first step at which train accuracy reaches 90% or 75%,
    using the saved learning curves (C3 keeps them for isolated runs)."""
    thr = {"t90": 0.90, "t75": 0.75}[metric]
    out = []
    for r in runs:
        if r["mode"] != "joint":
            if "curve" not in r:
                continue  # pilot isolated runs have no curve: cannot be re-timed
            c = r["curve"]
            r = {**r, "t95": next((st for st, a in zip(c["step"], c["train_acc"]) if a >= thr), None)}
        out.append(r)
    return out


def report(runs_paths, exps, out_path, n_seeds_label, metric="t95"):
    runs = sum((A.load(p) for p in runs_paths), [])
    if metric != "t95":
        runs = retime(runs, metric)
    rows = A.speed_table(runs)
    lines = [f"# Offset tests ({', '.join(exps)}; speed metric {metric.upper()}; generated by stats.py)\n",
             "delta = gap*(m) - gap*(2): how much faster (in log2 units of isolated learning time) the complex "
             "feature must be, relative to the linear control, before the network is indifferent. "
             "delta < 0 is a residual simplicity preference. "
             f"Bootstrap: {N_BOOT} resamples of seeds (joint and isolated runs) within cells. "
             f"SESOI = {SESOI} log2 units (a {2**SESOI:.2f}x speed ratio). p_holm: Holm-adjusted within each "
             "(experiment, depth) family of four tests. 'pooled' = mean offset over both optimisers and m = 3, 5.\n"]
    all_offs = {}
    for rule in ("main", "censoring_only"):
        ind = fits(rows, exps, rule)
        offs = offsets(ind, exps)
        all_offs[rule] = offs
        title = ("Main analysis (validity rule: isolated runs must reach 95% train and 95% test accuracy)"
                 if rule == "main" else "Sensitivity: validity rule dropped (only censored cells excluded)")
        lines += [f"## {title}\n",
                  "| exp | depth | optimiser | m | delta | 95% CI | 90% CI | p | p_holm | slope ratio b(m)/b(2) [95% CI] | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in offs:
            sr = (f"{f['slope_ratio']:.2f} {fmt_ci(f['slope_ci'])}" if f["ls_draws"] is not None else "")
            ph = "" if not np.isfinite(f["p_holm"]) else f"{f['p_holm']:.3f}"
            lines.append(f"| {f['exp']} | {f['depth']} | {f['opt']} | {f['m']} | {f['est']:.2f} | {fmt_ci(f['ci95'])} | "
                         f"{fmt_ci(f['ci90'])} | {f['p']:.3f} | {ph} | {sr} | {verdict(f)} |")
        lines.append("")
        if rule == "main":
            lines += ["### Per-m indifference points used above\n",
                      "| exp | optimiser | depth | m | gap* | 95% CI | slope | valid cells |", "|---|---|---|---|---|---|---|---|"]
            for k, r in sorted(ind.items()):
                lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {r['gap_star']:.2f} | [{r['lo']:.2f}, {r['hi']:.2f}] | "
                             f"{r['slope']:.2f} | {r['n_cond']} |")
            lines.append("")
    if n_seeds_label:
        lines += [f"## Sample-size projection from these data ({n_seeds_label} seeds per cell)\n",
                  "MDE = smallest offset detectable with 80% power at two-sided alpha = .05 with the current seeds. "
                  "'seeds for |delta| = x' = seeds per cell needed to detect an offset of size x, same grid. "
                  "Normal approximation from the bootstrap SE; not observed power.\n",
                  "| exp | depth | optimiser | m | bootstrap SE | MDE now | seeds for 0.3 | seeds for 0.5 |",
                  "|---|---|---|---|---|---|---|---|"]
        for f, mde, need in power_rows(all_offs["main"], int(n_seeds_label)):
            lines.append(f"| {f['exp']} | {f['depth']} | {f['opt']} | {f['m']} | {f['se']:.3f} | {mde:.2f} | {need[0.3]} | {need[0.5]} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as fh:
        fh.write(text)
    print(text)
    return all_offs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=[os.path.join(A.RES, "runs.jsonl")])
    ap.add_argument("--exps", default="E1,E3,R1")
    ap.add_argument("--out", default=os.path.join(A.RES, "stats_pilot.md"))
    ap.add_argument("--seeds", default="5", help="seeds per cell, for the sample-size projection ('' to skip)")
    ap.add_argument("--metric", default="t95", choices=["t95", "t90", "t75"],
                    help="speed metric for the isolated runs (t90/t75 need saved curves)")
    a = ap.parse_args()
    report(a.runs, a.exps.split(","), a.out, a.seeds, a.metric)
