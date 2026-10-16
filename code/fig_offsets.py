"""Figure 5: complexity offsets delta (gap*(m) - gap*(2)) with 95% bootstrap CIs,
for the pilot (exploratory) and the pre-registered confirmatory run C3 under
three speed metrics. Usage: python fig_offsets.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze as A
import stats as S

RES = A.RES
sets = [
    ("Pilot depth 1 (T95, exploratory)", "runs.jsonl", "E1", "t95", 1),
    ("Pilot depth 3 (T95, exploratory)", "runs.jsonl", "E3", "t95", 3),
    ("C3 depth 3 (T95, pre-registered primary)", "runs_confirm.jsonl", "C3", "t95", 3),
    ("C3 depth 3 (T90, pre-registered check)", "runs_confirm.jsonl", "C3", "t90", 3),
    ("C3 depth 3 (T75, pre-registered check)", "runs_confirm.jsonl", "C3", "t75", 3),
]
labels = [("sgd", "3"), ("sgd", "5"), ("adam", "3"), ("adam", "5"), ("pooled", "3,5")]

fig, ax = plt.subplots(figsize=(8.5, 6.5))
y, yt, ytl = 0, [], []
for title, fname, exp, metric, depth in sets:
    runs = A.load(os.path.join(RES, fname))
    if metric != "t95":
        runs = S.retime(runs, metric)
    rows = A.speed_table(runs)
    offs = {(f["opt"], f["m"]): f for f in S.offsets(S.fits(rows, [exp], "main"), [exp]) if f["depth"] == depth}
    ax.text(-1.55, y - 0.15, title, fontsize=9, weight="bold", va="bottom")
    y -= 0.6
    for opt, m in labels:
        f = offs[(opt, m)]
        pooled = opt == "pooled"
        ax.errorbar(f["est"], y, xerr=[[f["est"] - f["ci95"][0]], [f["ci95"][1] - f["est"]]],
                    fmt="D" if pooled else "o", color="k" if pooled else "#555", ms=6 if pooled else 4, capsize=2)
        yt.append(y)
        ytl.append(f"{opt}, m={m}" if not pooled else "pooled")
        y -= 0.45
    y -= 0.35
ax.axvline(0, color="k", lw=0.8)
ax.set_ylim(y + 0.2, 0.5)
ax.axvspan(-S.SESOI, S.SESOI, color="#2ca02c", alpha=0.08, label=f"+-{S.SESOI} (SESOI)")
ax.set_yticks(yt)
ax.set_yticklabels(ytl, fontsize=8)
ax.set_xlim(-1.6, 1.4)
ax.set_xlabel("complexity offset delta = gap*(m) - gap*(2)   (log2 units of isolated learning time)\n"
              "<- residual simplicity preference          anti-simplicity ->")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Is there a complexity effect beyond learning speed?  (95% bootstrap CIs)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(A.FIG, "fig5_offsets.png"), dpi=150, bbox_inches="tight")
print("saved")
