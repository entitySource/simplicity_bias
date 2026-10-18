# Simple, or just fast? A small study of simplicity bias

This folder holds a research proposal that tries to falsify, or qualify, the claim that neural networks preferentially learn simpler predictive features. It also holds the code and results of a small pilot study that implements the core experiments, a self-review of that pilot, and a pre-registered confirmatory run.

- **[proposal.md](proposal.md)**: the main deliverable (research question, hypotheses, design, controls, outcomes, limitations, results).
- **[preregistration.md](preregistration.md)**: the confirmatory depth-3 test, fixed before it was run.
- **[review/self_review.md](review/self_review.md)**: structured critique of the pilot and what changed in response.
- **code/**: everything needed to reproduce the pilot.
- **results/**: raw run records and summary tables.
- **figures/**: the three figures referenced in the proposal.

## The idea in one paragraph

In the standard demonstration (Shah et al., 2020), a network sees a linear feature and a k-slab feature that both predict the label perfectly, and it uses only the linear one. But at the default input scale the linear feature is also *learned much faster* when trained alone. Complexity (the number of slabs) does not change if you multiply a coordinate by a constant, while learning speed does. So I vary complexity and isolated learning speed independently and ask: once the two features are learned equally fast on their own, does the network still prefer the simpler one?

## Setup

- Data: 22-dimensional inputs. Coordinate 1 is the simple feature (linear threshold, 2 slabs), coordinate 2 is the complex feature (m slabs, m in {2, 3, 5}), coordinates 3 to 22 are uniform noise. Each feature can be rescaled (alpha_s, alpha_c) and made less predictive (p_s, p_c) without changing its marginal distribution.
- Model: ReLU MLP, width 128, 1 hidden layer (3 in E3), PyTorch default init.
- Training: cross-entropy, batch 128, SGD+momentum 0.9 at lr 0.05 or Adam at lr 1e-3, 4,000 steps (8,000 in E2). No tuning per condition.
- Outcome: P_s, the share of "conflict" test points (simple feature says class A, complex says class B) classified as A. Cross-check: accuracy drop when one coordinate is shuffled.
- Speed: T95, steps to 95% training accuracy when only one feature is predictive. gap = log2(T_complex / T_simple).

## Reproducing

Requires Python 3.10+, `torch`, `numpy`, `matplotlib`. Everything runs on CPU.

```bash
pip install -r requirements.txt
cd code
python run_all.py --procs 4 --seeds 5                  # pilot: 2,310 runs, about 1.5 hours on 4 CPU cores
SB_OUT=../results/runs_confirm.jsonl \
  python run_all.py --only C3 --seeds 10 --seed0 200   # confirmatory: 1,080 runs, about 80 minutes
python analyze.py                                      # tables in results/summary.md, figures 1-4
python stats.py                                        # pilot offset tests -> results/stats_pilot.md
python stats.py ../results/runs_confirm.jsonl --exps C3 --seeds "" --out ../results/stats_confirm.md
python stats.py ../results/runs_confirm.jsonl --exps C3 --seeds "" --metric t90 --out ../results/stats_confirm_t90.md
python stats.py ../results/runs_confirm.jsonl --exps C3 --seeds "" --metric t75 --out ../results/stats_confirm_t75.md
python fig_offsets.py                                  # figure 5
```

`run_all.py` appends to `results/runs.jsonl` (or the path in `SB_OUT`) and skips runs already present, so it can be stopped and resumed. `--only E1,E2` restricts the experiments. Each run is seeded (data and initialisation), so results should reproduce up to floating-point nondeterminism across machines.

## Files

| File | Contents |
|---|---|
| `code/sb.py` | Data generator (m-slab features, predictive strength, scales), MLP, training loop, metrics (T95, conflict preference, shuffle drops). |
| `code/run_all.py` | The experiment grid (E1, E2, E3 with follow-ups E3b/E3c, R1) and a resumable parallel runner. |
| `code/analyze.py` | Per-condition tables, indifference-point fits with seeded bootstrap CIs, figures 1 to 4. |
| `code/stats.py` | Offset tests (gap\*(m) − gap\*(2)), Holm correction, equivalence check, slope ratios, sensitivity to the validity rule, alternative speed metrics, sample-size projection. |
| `code/fig_offsets.py` | Figure 5: every offset estimate with its CI. |
| `results/runs_confirm.jsonl` | The 1,080 confirmatory runs (C3), with learning curves for isolated runs. |
| `results/stats_pilot.md`, `results/stats_confirm*.md` | Offset tests for the pilot and for C3 (T95, T90, T75). |
| `results/runs.jsonl` | One JSON record per training run (config, seed, metrics, thinned learning curves for joint runs). |
| `results/summary.md` | All tables generated from the runs. |
| `results/speed_vs_preference.csv` | Per-condition speeds and preferences for E1, E3, R1. |
| `figures/fig1_preference_vs_speed.png` | E1: preference vs isolated speed gap, by complexity and optimiser. |
| `figures/fig2_predictive_strength.png` | E2: preference vs predictive strength of the simple feature. |
| `figures/fig3_dynamics_predictive_strength.png` | E2: how preference evolves during training. |
| `figures/fig4_confirmatory_depth3.png` | C3: preference vs speed gap at depth 3, fresh seeds. |
| `figures/fig5_offsets.png` | All complexity offsets: pilot, C3, and C3 under alternative speed metrics. |

## Results in brief

- At natural scale the network uses only the linear feature (reproduces Shah et al.).
- Rescaling a coordinate, which leaves its complexity unchanged, flips the preference.
- 1 hidden layer: at matched isolated learning speed, complexity has no detectable effect. The pooled offset is 0.05 (log2 units of learning time), 90% CI [−0.07, 0.23], inside a ±0.5 equivalence bound. This analysis is exploratory.
- 3 hidden layers: a pre-registered rerun confirmed a small residual preference for the simpler feature, an offset of −0.21 [−0.29, −0.14]. But it turns into +0.16 [0.00, 0.25] when speed is measured at 90% instead of 95% training accuracy. Any complexity effect beyond speed is small, and its sign depends on the speed metric.
- Speed predicts *where* the preference flips but not how sharply: complex features switch over a narrower range.
- If the simple feature is wrong on 5% of examples, the network eventually switches to a 3-slab feature; for a 5-slab feature it partly memorises the exceptions instead.

Details, caveats and the full interpretation are in Sections 8, 10 and 11 of the proposal.

## Choices I made (and why)

- **Synthetic data only.** It is the only setting where complexity, scale and predictive strength can each be set exactly. A real-data version is proposed (E7) but not run.
- **Speed is measured in isolation**, not in the joint run, so the covariate is defined without looking at the outcome.
- **One learning rate per optimiser, no per-condition tuning**, because tuning would change learning speed, the very variable under study.
- **5 seeds** in the pilot to keep it to about an hour on a laptop. The confirmatory run uses 10, chosen from a power projection on the pilot's bootstrap standard errors (`results/stats_pilot.md`).
- **Exploratory vs confirmatory.** Only the depth-3 C3 test was pre-registered. Everything else is exploratory and labelled as such.
- **m = 7 was dropped** after piloting: a 7-slab feature alone did not generalise within the step budget (test accuracy near 0.6), so its absence in joint runs would not be informative.

## Tools

The self-review, power projection and pre-registration were structured with three open-source agent skills: `peer-review`, `statistical-power` and `hypothesis-generation`, from K-Dense's Scientific Agent Skills (github.com/K-Dense-AI/claude-scientific-skills). The causal-language check used that collection's `lint_causal_claims.py`. The skill authors ask to be cited as: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.* arXiv:2609.00065.
