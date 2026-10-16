# Pre-registration: confirmatory test of the depth-3 complexity offset (C3)

*Written 2026-10-15, after the pilot (E1, E2, E3, R1) and before any C3 run. The C3 grid is fixed in `code/run_all.py` (`C3_CELLS`). Nothing below may be changed after the C3 runs start; any deviation goes in the "Deviations" section at the end with its date and reason.*

## Why this test exists

Re-analysing the pilot with a direct test of the offset (`code/stats.py`, `results/stats_pilot.md`) changed the depth-3 picture. The original rule (do two separate CIs overlap?) gave "1 of 4 significant". Testing the difference itself gives 3 of 4 after Holm correction, and a pooled offset of δ = −0.43 [−0.60, −0.19]. That result is still not trustworthy as a confirmation, for three reasons:

1. **Cells were chosen after looking at data.** Four of the eight depth-3 cells per (optimiser, m), E3b, were added to bracket a crossover seen in the first E3 results.
2. **The control is thin.** The depth-3 m = 2 control rests on 3 cells, so gap\*(2) is weakly pinned down, and every offset is measured against it.
3. **The analysis was chosen after seeing the data.** The offset test, the pooled estimate and the SESOI below were all defined after the pilot results were known.

C3 repeats the depth-3 comparison on fresh seeds, with a grid fixed in advance and a control as dense as the treatment cells.

## Question

At depth 3 (three hidden layers, width 128, ReLU), once isolated learning speed is matched, does the network still prefer a linear feature over a 3-slab or 5-slab feature?

## Hypotheses (candidates, not findings)

| ID | Candidate | Prediction for the pooled offset δ in C3 |
|---|---|---|
| H_SB3 | A genuine residual complexity effect at depth 3 | δ < 0, 95% CI excludes 0, similar size to the pilot (about −0.4) |
| R_select | The pilot offset came from choosing cells after seeing data | δ shrinks towards 0; CI includes 0 |
| R_control | The pilot offset came from a poorly estimated control (3 cells) | gap\*(2) moves; δ shrinks towards 0 |
| R_metric | The offset is an artefact of the T95 speed metric | δ changes sign or loses significance when speed is measured as T90 or T75 |
| R_chance | Sampling noise | CI includes 0 |

These do not exclude each other: a real effect can be smaller than the pilot estimate because of selection.

## Design

- **Model and training:** identical to the pilot E3 runs: depth 3, width 128, ReLU, batch 128, 4,000 steps, SGD (momentum 0.9, lr 0.05) and Adam (lr 1e-3), no tuning.
- **Data:** identical generator (`code/sb.py`), p_s = p_c = 1, 20 noise dimensions.
- **Seeds:** 200 to 209 (10 per cell). No pilot seed is reused. Seed 100 was used once to time the code and is not part of C3.
- **Grid** (α_s, α_c), each run in joint, only_s and only_c mode:
  - m = 2: (0.5, 1), (0.7, 1), (1, 1), (1, 1.4), (1, 2), (1, 0.7)
  - m = 3: (1, 1.5), (1, 2), (1, 3), (1, 4), (1, 6), (1, 8)
  - m = 5: (1, 8), (1, 16), (0.5, 8), (0.4, 8), (0.3, 8), (0.5, 16)
- **Total:** 2 optimisers × 18 cells × 3 modes × 10 seeds = 1,080 runs.
- **Sample size rationale:** the pilot bootstrap SE of the pooled depth-3 offset was 0.108 with 5 seeds. With 10 seeds and a fixed grid this projects to about 0.076, a minimum detectable pooled offset of about 0.21 at 80% power and two-sided α = .05 (normal approximation; `results/stats_pilot.md`). Per-(optimiser, m) offsets are secondary and may be underpowered for SGD, m = 5 (projected MDE about 0.47).

## Primary analysis (fixed)

- **Estimand:** pooled offset δ = mean over {SGD, Adam} × {m = 3, m = 5} of gap\*(m) − gap\*(2).
  - gap\* is the speed gap where the fitted sigmoid P_s = sigmoid(b · (gap − gap\*)) crosses 0.5.
  - gap = log2(T_c / T_s), where T is the median steps to 95% train accuracy in the isolated runs.
- **Fit and CI:** `code/stats.py` unchanged, run with `--exps C3`.
  - Least-squares sigmoid on seed-level P_s.
  - 2,000 bootstrap resamples of seeds within cells, resampling joint and isolated runs.
  - Validity rule: both isolated runs reach 95% train accuracy and at least 95% test accuracy.
- **Decision rule:**
  - **Replicated:** the 95% CI of δ lies entirely below 0.
  - **Not replicated:** the 95% CI includes 0.
  - **Small if real:** additionally, if the 90% CI lies within ±0.5 (the SESOI, a 1.41× speed ratio), the effect is described as small.
  - A 95% CI entirely above 0 would be an anti-simplicity offset.

## Secondary analyses (fixed, reported whatever they show)

1. Per-(optimiser, m) offsets with Holm correction across the four tests.
2. Slope ratio b(m)/b(2) with bootstrap CI. H_speed predicts a ratio of 1. The pilot already showed ratios above 1 at depth 1, so this is a known open issue, not a new test.
3. **Speed-metric robustness (R_metric):** recompute δ with T90 and T75 in place of T95, from the isolated-run learning curves that C3 now saves.
4. The same analysis with the validity rule dropped.
5. Pilot E3 and C3 combined, labelled as exploratory because the pilot part is not confirmatory.

## What stays exploratory

Everything in the pilot is exploratory, including the depth-1 equivalence result (pooled δ = 0.05, 90% CI [−0.07, 0.23]). That result was obtained with an analysis defined after the data were seen. It is reported as such, not as a confirmed null.

## Deviations

- **2026-10-15, before any C3 result existed.** The first C3 launch was stopped after about a minute, with 0 runs written. The script was then changed to keep every evaluated learning-curve step below step 500 for C3 isolated runs; it had previously kept every 5th. The reason was to make the pre-registered T90/T75 check usable. The change only affects stored curves, not training or any metric, so its expected impact on the primary analysis is none.
- **Afterwards, none.** The primary and secondary analyses were run exactly as written above. No cell failed the validity rule.
