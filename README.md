# Simple, or just fast? A small study of simplicity bias

This folder holds a research proposal that tries to falsify, or qualify, the claim that neural networks preferentially learn simpler predictive features, plus the code and results of a small pilot study that implements its core experiments.

- **[proposal.md](proposal.md)**: the main deliverable (research question, hypotheses, design, controls, outcomes, limitations, pilot results).
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
pip install torch numpy matplotlib
cd code
python run_all.py --procs 4 --seeds 5      # 2,310 runs; about 1.5 hours on 4 CPU cores
python analyze.py ../results/runs.jsonl    # writes results/summary.md, results/*.csv, figures/*.png
```

`run_all.py` appends to `results/runs.jsonl` (or the path in `SB_OUT`) and skips runs already present, so it can be stopped and resumed. `--only E1,E2` restricts the experiments. Each run is seeded (data and initialisation), so results should reproduce up to floating-point nondeterminism across machines.

## Files

| File | Contents |
|---|---|
| `code/sb.py` | Data generator (m-slab features, predictive strength, scales), MLP, training loop, metrics (T95, conflict preference, shuffle drops). |
| `code/run_all.py` | The experiment grid (E1, E2, E3 with follow-ups E3b/E3c, R1) and a resumable parallel runner. |
| `code/analyze.py` | Per-condition tables, indifference-point fits with bootstrap CIs, figures. |
| `results/runs.jsonl` | One JSON record per training run (config, seed, metrics, thinned learning curves for joint runs). |
| `results/summary.md` | All tables generated from the runs. |
| `results/speed_vs_preference.csv` | Per-condition speeds and preferences for E1, E3, R1. |
| `figures/fig1_preference_vs_speed.png` | E1: preference vs isolated speed gap, by complexity and optimiser. |
| `figures/fig2_predictive_strength.png` | E2: preference vs predictive strength of the simple feature. |
| `figures/fig3_dynamics_predictive_strength.png` | E2: how preference evolves during training. |

## Results in brief

- At natural scale the network uses only the linear feature (reproduces Shah et al.).
- Rescaling a coordinate, which leaves its complexity unchanged, flips the preference.
- 1 hidden layer: at matched isolated learning speed, complexity has no detectable effect (null not rejected, SGD and Adam).
- 3 hidden layers: a small residual preference for the simpler feature, significant in 1 of 4 cases (suggestive only).
- If the simple feature is wrong on 5% of examples, the network eventually switches to a 3-slab feature; for a 5-slab feature it partly memorises the exceptions instead.

Details, caveats and the full interpretation are in Section 8 and 11 of the proposal.

## Choices I made (and why)

- **Synthetic data only.** It is the only setting where complexity, scale and predictive strength can each be set exactly. A real-data version is proposed (E7) but not run.
- **Speed is measured in isolation**, not in the joint run, so the covariate is defined without looking at the outcome.
- **One learning rate per optimiser, no per-condition tuning**, because tuning would change learning speed, the very variable under study.
- **5 seeds** in the pilot to keep it to about an hour on a laptop. The proposal recommends 20 for the full study.
- **m = 7 was dropped** after piloting: a 7-slab feature alone did not generalise within the step budget (test accuracy near 0.6), so its absence in joint runs would not be informative.
