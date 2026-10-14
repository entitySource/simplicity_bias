"""Run the implemented experiment suite and append one JSON line per run.

Usage: python run_all.py [--procs 4] [--seeds 5] [--only E1,E2,...]
Results go to ../results/runs.jsonl (existing runs are skipped, so the
script can be resumed).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from multiprocessing import Pool

import torch

from sb import DataCfg, TrainCfg, train

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SB_OUT", os.path.join(HERE, "..", "results", "runs.jsonl"))


def jobs(seeds: int, only: set[str]):
    js = []
    # E1: decouple complexity (m) from learning speed (relative coordinate scale)
    if "E1" in only:
        for opt, m, a_s, a_c, mode, s in itertools.product(
            ["sgd", "adam"], [2, 3, 5], [1, 0.3, 0.1, 0.03], [1, 2, 4, 8],
            ["only_s", "only_c", "joint"], range(seeds)):
            js.append(("E1", dict(m=m, alpha_s=a_s, alpha_c=a_c), dict(opt=opt), mode, s))
    # E2: predictive strength of the simple feature vs complexity (natural scale)
    if "E2" in only:
        for opt, m, p_s, s in itertools.product(
            ["sgd", "adam"], [3, 5], [1.0, 0.95, 0.9, 0.8, 0.7, 0.6], range(seeds)):
            js.append(("E2", dict(m=m, p_s=p_s), dict(opt=opt, steps=8000), "joint", s))
    # E3: architecture (depth 3) at a subset of E1 settings
    if "E3" in only:
        for opt, m, (a_s, a_c), mode, s in itertools.product(
            ["sgd", "adam"], [3, 5], [(1, 1), (1, 4), (1, 8), (0.1, 8)],
            ["only_s", "only_c", "joint"], range(seeds)):
            js.append(("E3", dict(m=m, alpha_s=a_s, alpha_c=a_c), dict(opt=opt, depth=3), mode, s))
    # E3b: extra depth-3 cells added after E3, to bracket the crossover that the
    # first E3 grid left between two distant points
    if "E3b" in only:
        for opt, m, (a_s, a_c), mode, s in itertools.product(
            ["sgd", "adam"], [3, 5], [(0.3, 8), (0.5, 8), (0.3, 4), (1, 16)],
            ["only_s", "only_c", "joint"], range(seeds)):
            js.append(("E3", dict(m=m, alpha_s=a_s, alpha_c=a_c), dict(opt=opt, depth=3), mode, s))
    # E3c: symmetry control (m = 2) for depth 3
    if "E3c" in only:
        for opt, (a_s, a_c), mode, s in itertools.product(
            ["sgd", "adam"], [(1, 1), (1, 2), (0.5, 1)], ["only_s", "only_c", "joint"], range(seeds)):
            js.append(("E3", dict(m=2, alpha_s=a_s, alpha_c=a_c), dict(opt=opt, depth=3), mode, s))
    # R1: robustness, no noise coordinates (scale then cannot change SNR vs noise)
    if "R1" in only:
        for m, (a_s, a_c), mode, s in itertools.product(
            [3, 5], [(1, 1), (1, 4), (1, 8), (0.3, 8), (0.1, 8), (0.1, 1)],
            ["only_s", "only_c", "joint"], range(seeds)):
            js.append(("R1", dict(m=m, alpha_s=a_s, alpha_c=a_c, d_noise=0), dict(opt="sgd"), mode, s))
    return js


def key(j):
    return json.dumps([j[0], j[1], j[2], j[3], j[4]], sort_keys=True)


def run(j):
    torch.set_num_threads(1)
    exp, dkw, tkw, mode, seed = j
    r = train(DataCfg(**dkw), TrainCfg(**tkw), seed, mode)
    if mode != "joint":
        r.pop("curve")
    else:  # keep the curve but thin it
        c = r["curve"]
        keep = [i for i, st in enumerate(c["step"]) if st < 100 and st % 5 == 0 or st % 50 == 0]
        r["curve"] = {k: [v[i] for i in keep] for k, v in c.items()}
    return {"key": key(j), "exp": exp, "data": dkw, "train": tkw, "mode": mode, "seed": seed, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="E1,E2,E3,E3b,E3c,R1")
    a = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [j for j in jobs(a.seeds, set(a.only.split(","))) if key(j) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already done)", flush=True)
    with Pool(a.procs) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run, todo, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
