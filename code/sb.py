"""Core of the simplicity-bias study: synthetic data, model, training, metrics.

Data ("linear + m-slab", after Shah et al. 2020, LMS):
  x = [alpha_s * f_s, alpha_c * f_c, noise_1..noise_d]
  f_s : "simple" coordinate, a 2-slab (linear threshold at 0) feature.
  f_c : "complex" coordinate, an m-slab feature (m-1 decision boundaries).
        m = 2 makes f_c statistically identical to f_s (symmetry control).
  noise: d coordinates ~ U[-1, 1], independent of the label.
Every slab lives in [-1, 1]; adjacent slabs are separated by a gap of 2*GAP,
so the minimum margin is the same for every m.

Predictive strength: with probability p a feature is drawn from the slabs of
the true class y, otherwise from the slabs of -y. The marginal distribution
of the feature does not change with p; only its agreement with y does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

GAP = 0.05


def slab_feature(labels: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Sample an m-slab coordinate whose slab class matches `labels` (+-1)."""
    edges = np.linspace(-1.0, 1.0, m + 1)
    # slab i has class +1 when (m-1-i) is even; slabs alternate classes
    cls = np.array([1 if (m - 1 - i) % 2 == 0 else -1 for i in range(m)])
    pos, neg = np.where(cls == 1)[0], np.where(cls == -1)[0]
    out = np.empty(len(labels))
    for lab, idx in ((1, pos), (-1, neg)):
        sel = np.where(labels == lab)[0]
        slab = rng.choice(idx, size=len(sel))
        lo, hi = edges[slab] + GAP, edges[slab + 1] - GAP
        out[sel] = rng.uniform(lo, hi)
    return out


def corrupt(labels: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of labels where each entry is flipped with prob 1-p."""
    flip = rng.random(len(labels)) > p
    return np.where(flip, -labels, labels)


@dataclass
class DataCfg:
    m: int = 3            # slabs in the complex coordinate
    alpha_s: float = 1.0  # scale of simple coordinate
    alpha_c: float = 1.0  # scale of complex coordinate
    p_s: float = 1.0      # agreement of simple coordinate with label
    p_c: float = 1.0      # agreement of complex coordinate with label
    d_noise: int = 20
    n: int = 4000


def make_data(cfg: DataCfg, rng: np.random.Generator, mode: str = "joint"):
    """mode: 'joint' (both features carry signal), 'only_s', 'only_c'.

    In the 'only_*' modes the other coordinate keeps its marginal
    distribution but is drawn from an independent random label, so it is
    present but uninformative.
    """
    y = rng.choice([-1, 1], size=cfg.n)
    ys = corrupt(y, cfg.p_s, rng)
    yc = corrupt(y, cfg.p_c, rng)
    if mode == "only_s":
        yc = rng.choice([-1, 1], size=cfg.n)
    elif mode == "only_c":
        ys = rng.choice([-1, 1], size=cfg.n)
    fs = slab_feature(ys, 2, rng)
    fc = slab_feature(yc, cfg.m, rng)
    noise = rng.uniform(-1, 1, size=(cfg.n, cfg.d_noise))
    x = np.column_stack([cfg.alpha_s * fs, cfg.alpha_c * fc, noise])
    return torch.tensor(x, dtype=torch.float32), torch.tensor((y + 1) // 2, dtype=torch.long)


def make_conflict(cfg: DataCfg, rng: np.random.Generator, n: int = 4000):
    """Test points where the two features point to opposite classes.

    Returns x and the label implied by the simple feature. Each coordinate is
    still drawn from its own training marginal; only the pairing is new.
    """
    ys = rng.choice([-1, 1], size=n)
    fs = slab_feature(ys, 2, rng)
    fc = slab_feature(-ys, cfg.m, rng)
    noise = rng.uniform(-1, 1, size=(n, cfg.d_noise))
    x = np.column_stack([cfg.alpha_s * fs, cfg.alpha_c * fc, noise])
    return torch.tensor(x, dtype=torch.float32), torch.tensor((ys + 1) // 2, dtype=torch.long)


def mlp(d_in: int, width: int, depth: int, act: str = "relu") -> nn.Module:
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh}
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), acts[act]()]
        d = width
    layers.append(nn.Linear(d, 2))
    return nn.Sequential(*layers)


@dataclass
class TrainCfg:
    opt: str = "sgd"       # 'sgd' (momentum 0.9) or 'adam'
    lr: float | None = None
    depth: int = 1
    width: int = 128
    act: str = "relu"
    steps: int = 4000
    batch: int = 128


DEFAULT_LR = {"sgd": 0.05, "adam": 1e-3}


@torch.no_grad()
def acc(model, x, y) -> float:
    return (model(x).argmax(1) == y).float().mean().item()


def train(dcfg: DataCfg, tcfg: TrainCfg, seed: int, mode: str = "joint") -> dict:
    """Train one network; return learning curves and final reliance metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr = make_data(dcfg, rng, mode)
    test_cfg = DataCfg(**{**asdict(dcfg), "p_s": 1.0, "p_c": 1.0})
    xte, yte = make_data(test_cfg, rng, "joint")
    xcf, ycf_s = make_conflict(dcfg, rng)

    model = mlp(xtr.shape[1], tcfg.width, tcfg.depth, tcfg.act)
    lr = tcfg.lr or DEFAULT_LR[tcfg.opt]
    if tcfg.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    # dense evaluation early (to time fast features), sparse later
    eval_at = set(range(0, 100)) | set(range(100, 500, 5)) | set(range(500, tcfg.steps + 1, 25))
    curve = {"step": [], "train_acc": [], "p_simple": []}
    t95 = None
    n = len(xtr)
    for step in range(tcfg.steps + 1):
        if step in eval_at or step == tcfg.steps:
            tr = acc(model, xtr, ytr)
            curve["step"].append(step)
            curve["train_acc"].append(tr)
            curve["p_simple"].append(acc(model, xcf, ycf_s))
            if t95 is None and tr >= 0.95:
                t95 = step
        if step == tcfg.steps:
            break
        idx = torch.randint(0, n, (tcfg.batch,))
        opt.zero_grad()
        lossf(model(xtr[idx]), ytr[idx]).backward()
        opt.step()

    # randomization test: shuffle one coordinate across the clean test set
    def shuffled(col):
        xs = xte.clone()
        xs[:, col] = xs[torch.randperm(len(xs)), col]
        return acc(model, xs, yte)

    return {
        "t95": t95,
        "final_train_acc": curve["train_acc"][-1],
        "test_acc": acc(model, xte, yte),
        "acc_shuffle_s": shuffled(0),
        "acc_shuffle_c": shuffled(1),
        "p_simple": curve["p_simple"][-1],
        "curve": curve,
    }
