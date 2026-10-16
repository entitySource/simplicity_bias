# Self-review of the pilot (working draft)

*Structured with the `peer-review`, `statistical-power` and `hypothesis-generation` skills (K-Dense Scientific Agent Skills). This is the author's own review of their own work, so there are no confidentiality or conflict issues. Claude drafted it as an AI assistant; the author should check every point before relying on it.*

**Material reviewed:** `proposal.md` (version of 2026-10-13), `README.md`, `code/`, `results/summary.md` and `results/runs.jsonl` (2,310 runs).

## 1. Neutral map of the study

| Item | Content |
|---|---|
| Question | Does feature complexity (number of linear pieces) predict which feature a small MLP relies on, once isolated learning speed is controlled? |
| System | Synthetic LMS-style data: linear coordinate vs m-slab coordinate (m = 2, 3, 5) plus 20 noise coordinates |
| Unit of inference | Seed (data draw plus initialisation), nested in cells (m, α_s, α_c, optimiser, depth) |
| Manipulation | Coordinate scale α_s, α_c (changes speed, not complexity); m; optimiser; depth; predictive strength p_s |
| Outcome | Conflict preference P_s; shuffle-drop cross-check |
| Key estimand | Indifference point gap\* per m, and its offset from the m = 2 control |
| Principal claims | (C1) the natural-scale preference reproduces; (C2) rescaling flips it; (C3) at depth 1, complexity adds nothing beyond speed; (C4) at depth 3 a small residual simplicity preference, "1 of 4 significant"; (C5) 5% label noise on the simple feature flips the network to a 3-slab feature |

## 2. Claim–evidence check

| Claim | Evidence | Alignment | Action |
|---|---|---|---|
| C1 | E1 natural-scale cells, P_s = 1.000, shuffle drops 0.50 / 0.00 | Supported | None |
| C2 | E1 scale sweep; P_s from 1.00 to 0.03 within fixed m | Supported, large effect | None |
| C3 | Overlap of per-m CIs at depth 1 | **Over-stated.** "No detectable effect" is shown only as non-rejection; no equivalence bound was stated | Test the offset directly and add an equivalence bound (M1, M2) |
| C4 | Overlap of per-m CIs at depth 3 | **Under-tested and post hoc.** Wrong test (CI overlap), no multiplicity correction, cells chosen after seeing data, thin control | Direct offset test with Holm correction, then a pre-registered fresh-seed replication (M1, M3, M4) |
| C5 | E2 table | Supported for m = 3. For m = 5 the "memorises exceptions" reading rests on clean-test accuracy and shuffle drops, not on a direct memorisation measure | Soften the wording, or add a direct check (m5) |

## 3. Major comments

**M1. The H0 decision rule tests the wrong quantity.**
- *Location:* proposal §6.4 and §8.3, "rejected if the CI for gap\*(m) lies entirely below the CI for gap\*(2)".
- *Observation:* asking whether two 95% CIs overlap is much more conservative than testing their difference. Two estimates can differ significantly while their CIs overlap.
- *Why it matters:* the hypotheses are about the offset δ(m) = gap\*(m) − gap\*(2), so the inference should be on δ. The overlap rule biases the study towards "not rejected", which is the outcome the author's working hypothesis predicts.
- *Action:* bootstrap the difference directly and report its CI and p-value.

**M2. A null result is presented as evidence of no effect.**
- *Location:* §0 point 3, §8.3, §11 ("complexity has no detectable effect").
- *Observation:* a non-significant test does not establish equivalence. No smallest effect size of interest was defined.
- *Action:* state a SESOI and report whether the 90% CI of δ lies inside it (a TOST-style equivalence check). Label it exploratory, because the SESOI is chosen after the pilot.

**M3. Multiple comparisons at depth 3 are not handled.**
- *Location:* §8.5, "rejected in one of the four cases".
- *Observation:* there are four tests (2 optimisers × 2 complexities) and no family-wise correction. Counting significant tests is also a weak summary.
- *Action:* apply Holm correction within the family, and add one pooled primary estimate per depth.

**M4. The depth-3 evidence is exploratory but reads as a result.**
- *Location:* §8.1 (E3b, E3c added after the first look) and §8.5.
- *Observations:*
  - Half of the depth-3 cells were chosen to bracket a crossover already seen in the data.
  - The m = 2 control has 3 valid cells, so the reference point every offset is measured against is weakly estimated.
  - 5 seeds per cell.
- *Action:* pre-register a confirmatory replication: fresh seeds, a grid fixed in advance, a control as dense as the treatment cells, and a sample size justified by a power projection.

**M5. The "speed is a sufficient statistic" prediction is not tested.**
- *Location:* §5, "H_speed additionally predicts that the logistic curves for different m have similar slopes".
- *Observation:* this prediction is stated but never checked. Fitted slopes differ (b = 1.30 for the control vs 2.08 for m = 5, SGD).
- *Action:* report the slope ratio b(m)/b(2) with a CI, and say plainly if H_speed fails on this point.

**M6. The speed metric is a single, arbitrary threshold.**
- *Location:* limitation 2 in §10.
- *Observation:* the conclusions might depend on T95.
- *Action:* save the isolated-run learning curves and recompute δ with T90 and T75.

**M7. The validity rule was introduced after seeing data.**
- *Location:* §8.1.
- *Action:* report all inferential results with and without the rule.

**M8. Claim types are mixed.**
- *Location:* §0, §8.3 and §11, in phrases such as "explained by learning speed" and "speed, not complexity, is what places…".
- *Observation:* rescaling is an intervention, so "rescaling flips the preference" is a causal claim the design supports. "Speed explains the preference" is different. Isolated speed is *measured*, not set, and it changes together with scale, so this claim is predictive or associational across cells. It does not establish speed as the mechanism.
- *Evidence:* the `lint_causal_claims.py` tool in the hypothesis-generation skill flagged 7 lines of unmarked causal language (lines 19, 44, 82, 89, 91, 144, 205). Lexical flags still need judging by hand.
- *Action:* state the claim type at each of these places, for example "predicts" rather than "explains". Name the mechanism (gradient magnitude, gradient starvation) as a candidate, not a finding.

## 4. Minor comments

- **m1.** The R1 CIs quoted in §8.4 ([0.01, 0.65] and [−0.43, 1.02]) do not match `results/summary.md` ([0.01, 0.61] and [−0.25, 1.04]). Cause: the bootstrap used one global random stream, so its numbers depended on call order. Fix: seed the bootstrap per fit, regenerate, and quote only regenerated numbers.
- **m2.** There is no `requirements.txt` with versions. Record the torch, numpy and matplotlib versions used.
- **m3.** Figure 1 does not show the depth-3 data, which carry the most interesting claim.
- **m4.** §2 cites several papers "from memory". Each reference should be verified against the published version before submission.
- **m5.** The E2 claim that the 5-slab network "memorises the exceptions" should cite the evidence directly (train accuracy 0.99 to 1.00, clean test 0.91 to 0.96), or be softened to "fits the exceptions without switching features".
- **m6.** Seeds per cell (5) are not justified anywhere. Add the power projection.

## 5. Responses (what was done)

Each comment is answered in `proposal.md` §12. Main outcomes:

- **M1–M3.** The pilot's depth-3 offset became stronger under the correct test: pooled −0.43 [−0.60, −0.19]. At depth 1 the offset stayed at about zero: 0.05, 90% CI [−0.07, 0.23].
- **M4.** The pre-registered rerun replicated a depth-3 offset of −0.21 [−0.29, −0.14], about half the pilot estimate.
- **M6.** The offset changes sign with the speed metric (T90: +0.16 [0.00, 0.25]). The depth-3 "complexity effect" is therefore bounded but not identified.
- **M5.** The slope prediction of H_speed fails; complex features switch more sharply.
- **M7.** Dropping the validity rule changes nothing.
