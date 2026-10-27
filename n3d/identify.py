"""Identification metrics for generated clouds: does each output pick out its own object?

Mean Chamfer rewards hedging: a blurry average shape scores well against every target, so a
model can lower mean CD while carrying no information about the input. These scores cannot
be gamed that way. For every generated cloud, compute CD to every ground truth in the split
and ask where its own object ranks.

  inst_top1 / inst_top5   own object is the nearest / among the 5 nearest GTs (chance 1/N, 5/N)
  cat_top1                nearest GT is from the right category (chance ~ objects per cat / N)
  afc_within_cat          closer to own GT than to a same-category sibling's     (chance 0.5)
  afc_across_cat          closer to own GT than to a random other-category GT     (chance 0.5)
  mean_rank               mean rank of own GT, 1 = best (chance (N+1)/2)
Each is averaged over samples; p-values are one-sided binomial against chance on sample 0.

    python -m n3d.identify --run pn_film --file pred_test_k5_s100.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_bundle
from .encoder import run_dir


@torch.no_grad()
def cd_matrix(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 24) -> torch.Tensor:
    """pred (P,n,3), gt (G,n,3) -> (P,G) symmetric mean squared Chamfer."""
    out = torch.empty(len(pred), len(gt), device=pred.device)
    for i, p in enumerate(pred):
        for j in range(0, len(gt), chunk):
            d = torch.cdist(p[None].expand(len(gt[j:j + chunk]), -1, -1), gt[j:j + chunk]) ** 2
            out[i, j:j + chunk] = d.amin(2).mean(1) + d.amin(1).mean(1)
    return out


def binom_p(k: int, n: int, p0: float) -> float:
    from scipy.stats import binomtest
    return float(binomtest(k, n, p0, alternative="greater").pvalue)


def identify(D: np.ndarray, cats: np.ndarray, rng) -> dict:
    """D (N,N): row i = CD from generated cloud of object i to every GT. Returns hit arrays."""
    N = len(D)
    rank = (D < D[np.arange(N), np.arange(N)][:, None]).sum(1) + 1
    nearest = D.argmin(1)
    within, across = [], []
    for i in range(N):
        sib = np.flatnonzero((cats == cats[i]) & (np.arange(N) != i))
        if len(sib):
            within.append(D[i, i] < D[i, rng.choice(sib)])
        across.append(D[i, i] < D[i, rng.choice(np.flatnonzero(cats != cats[i]))])
    return {"rank": rank, "top1": rank == 1, "top5": rank <= 5, "cat": cats[nearest] == cats,
            "within": np.array(within), "across": np.array(across)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--file", default="pred_test_k5_s100.npz")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--points", type=int, default=2048, help="subsample both clouds to this many points")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    z = np.load(run_dir(args.run) / args.file)
    pred = z["pred"] if z["pred"].ndim == 4 else z["pred"][:, None]
    objs = z["obj"]
    b = load_bundle(args.data_root, ("sub01",), "point_cloud_simple")
    cats = b.obj_cat[objs]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    sub = lambda a: a[:, rng.choice(a.shape[1], min(args.points, a.shape[1]), replace=False)]
    G = torch.from_numpy(sub(z["gt"])).float().to(dev)
    N, K = pred.shape[:2]
    hits = {k: [] for k in ("rank", "top1", "top5", "cat", "within", "across")}
    for k in range(K):
        D = cd_matrix(torch.from_numpy(sub(pred[:, k])).float().to(dev), G).cpu().numpy()
        for key, v in identify(D, cats, rng).items():
            hits[key].append(v)
    per_cat = np.mean([np.sum(cats == c) for c in cats])
    chance = {"inst_top1": 1 / N, "inst_top5": min(1, 5 / N), "cat_top1": per_cat / N,
              "afc_within_cat": 0.5, "afc_across_cat": 0.5}
    keymap = {"inst_top1": "top1", "inst_top5": "top5", "cat_top1": "cat",
              "afc_within_cat": "within", "afc_across_cat": "across"}
    summary = {"file": args.file, "n_objects": int(N), "samples": int(K), "points": args.points,
               "mean_rank": float(np.mean(hits["rank"])), "mean_rank_chance": (N + 1) / 2}
    for name, key in keymap.items():
        h = np.concatenate(hits[key]).astype(bool)
        # samples of one object are not independent, so the p-value uses sample 0 only (n = objects)
        first = np.asarray(hits[key][0]).astype(bool)
        summary[name] = float(h.mean())
        summary[name + "_chance"] = float(chance[name])
        summary[name + "_p"] = binom_p(int(first.sum()), len(first), chance[name])
    out = run_dir(args.run) / f"identify_{Path(args.file).stem}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
