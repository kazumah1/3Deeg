"""Does the EEG predict coarse primitive geometry, and beyond what its category signal implies?

This is the go / no-go test for switching the output to primitives (Branch C). Primitives
lower the output dimension; they cannot add information the EEG does not carry. So before
building a primitive decoder, check whether low-dimensional primitive parameters are
linearly predictable from the EEG at all.

Targets (per object, in the pc_norm frame, rotation- and permutation-invariant):
  k=1 PCA-aligned oriented cuboid (same fit as benchmark/scripts/benchmark_k_oriented_cuboids.py):
      sorted extents e1>=e2>=e3, log(e2/e1), log(e3/e1)
  k=4 cuboids (deterministic farthest-point init + Lloyd k-means, as in the benchmark):
      sorted volume fractions (4), mean pairwise distance between component centres
Predictors, all scored on repeat-averaged test EEG with out-of-sample R^2 against the
training mean (R^2 <= 0 means no better than predicting the average shape):
  eeg_ridge         ridge regression from time-binned EEG (alpha picked on val)
  eeg_via_category  linear category classifier's probabilities x per-category mean targets
  oracle_category   true category's training-mean target (ceiling for category-level information)
  null              eeg_ridge refit on training targets shuffled across objects (permutations)
Decision rule: primitives are worth building only if eeg_ridge clears the null's 95th
percentile. If it does but not eeg_via_category, the signal is just category again.

    python -m n3d.primprobe --run prim_multi5 --participants sub01 sub02 sub03 sub04 sub05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .data import load_bundle, make_splits
from .encoder import run_dir
from .probe import averaged, bin_means, zscore_per_subject

TARGETS = ["e1", "e2", "e3", "log_e2_e1", "log_e3_e1", "k4_vol1", "k4_vol2", "k4_vol3", "k4_vol4", "k4_spread"]


def obb(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = xyz.mean(0)
    _, axes = np.linalg.eigh(np.cov((xyz - c).T))
    local = (xyz - c) @ axes[:, ::-1]
    lo, hi = local.min(0), local.max(0)
    return c + ((lo + hi) / 2) @ axes[:, ::-1].T, hi - lo


def kmeans(xyz: np.ndarray, k: int, iters: int = 100) -> np.ndarray:
    centers = [xyz[np.argmax(((xyz - xyz.mean(0)) ** 2).sum(1))]]
    near = ((xyz - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        centers.append(xyz[np.argmax(near)])
        near = np.minimum(near, ((xyz - centers[-1]) ** 2).sum(1))
    C = np.stack(centers)
    for _ in range(iters):
        a = ((xyz[:, None] - C[None]) ** 2).sum(2).argmin(1)
        new = np.stack([xyz[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.abs(new - C).max() < 1e-8:
            break
        C = new
    return a


def primitive_targets(points: np.ndarray, k: int = 4) -> np.ndarray:
    rows = []
    for xyz in points.astype(np.float64):
        _, ext = obb(xyz)
        e = np.sort(ext)[::-1]
        a = kmeans(xyz[::2], k)
        sub = xyz[::2]
        comps = [obb(sub[a == j]) if (a == j).sum() > 3 else (sub.mean(0), np.zeros(3)) for j in range(k)]
        vol = np.array([np.prod(np.maximum(x[1], 1e-6)) for x in comps])
        cen = np.stack([x[0] for x in comps])
        spread = np.mean([np.linalg.norm(cen[i] - cen[j]) for i in range(k) for j in range(i + 1, k)])
        rows.append([*e, np.log(e[1] / e[0]), np.log(e[2] / e[0]), *np.sort(vol / vol.sum())[::-1], spread])
    return np.array(rows)


def r2_oos(pred: np.ndarray, y: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    return 1 - ((y - pred) ** 2).sum(0) / ((y - train_mean) ** 2).sum(0)


def main(argv=None):
    from sklearn.linear_model import LogisticRegression, Ridge
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--geometry", default="point_cloud_simple")
    ap.add_argument("--split", default="ours", choices=list(config.SPLITS))
    ap.add_argument("--dyn-bins", type=int, default=30)
    ap.add_argument("--stat-bins", type=int, default=25)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1e2, 1e3, 1e4, 3e4, 1e5, 3e5])
    ap.add_argument("--C", type=float, default=1e-3, help="category classifier for eeg_via_category")
    ap.add_argument("--permutations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    s = make_splits(b, args.split)
    cache = run_dir(args.run) / "primitive_targets.npy"
    if cache.exists():
        T = np.load(cache)
    else:
        print("fitting primitives to all objects ...")
        T = primitive_targets(b.points)
        np.save(cache, T)
    X = np.concatenate([bin_means(b.dyn, args.dyn_bins), bin_means(b.stat, args.stat_bins)], 1).astype(np.float32)
    X = zscore_per_subject(X, b, s["train"])
    Xtr, ctr, _, otr = averaged(X, b, s["train"])
    Xva, _, _, ova = averaged(X, b, s["val"])
    Xte, cte, ste, ote = averaged(X, b, s["test"])
    mu, sd = T[s["train"]].mean(0), T[s["train"]].std(0) + 1e-9
    Z = (T - mu) / sd  # standardized targets; R^2 is unaffected
    Ytr, Yva, Yte = Z[otr], Z[ova], Z[ote]
    zero = np.zeros(Z.shape[1])

    best = max(args.alphas, key=lambda a: r2_oos(Ridge(alpha=a).fit(Xtr, Ytr).predict(Xva), Yva, zero).mean())
    ridge = Ridge(alpha=best).fit(Xtr, Ytr)
    res = {"eeg_ridge": r2_oos(ridge.predict(Xte), Yte, zero)}

    K = len(b.categories)
    cat_mean = np.zeros((K, Z.shape[1]))
    for c in range(K):
        m = b.obj_cat[s["train"]] == c
        if m.any():
            cat_mean[c] = Z[s["train"]][m].mean(0)
    clf = LogisticRegression(C=args.C, max_iter=2000).fit(Xtr, ctr)
    P = np.zeros((len(Xte), K)); P[:, clf.classes_] = clf.predict_proba(Xte)
    res["eeg_via_category"] = r2_oos(P @ cat_mean, Yte, zero)
    res["oracle_category"] = r2_oos(cat_mean[cte], Yte, zero)

    null = []
    for _ in range(args.permutations):
        perm = rng.permutation(len(Ytr))
        null.append(r2_oos(Ridge(alpha=best).fit(Xtr, Ytr[perm]).predict(Xte), Yte, zero).mean())
    null = np.array(null)
    p_val = float((1 + (null >= res["eeg_ridge"].mean()).sum()) / (1 + len(null)))

    out = {"alpha": best, "n_test_rows": int(len(Xte)), "targets": TARGETS,
           "mean_r2": {k: float(v.mean()) for k, v in res.items()},
           "per_target_r2": {k: dict(zip(TARGETS, map(float, v))) for k, v in res.items()},
           "null_mean_r2": float(null.mean()), "null_p95_r2": float(np.percentile(null, 95)), "p_eeg_ridge_vs_null": p_val,
           "per_subject_eeg_ridge": {args.participants[u]: float(r2_oos(ridge.predict(Xte[ste == u]), Yte[ste == u], zero).mean())
                                     for u in np.unique(ste)}}
    (run_dir(args.run) / "primitive_probe.json").write_text(json.dumps(out, indent=1))
    print(f"ridge alpha {best:g}; test rows {len(Xte)}")
    print(f"{'target':12s} " + " ".join(f"{k:>17s}" for k in res))
    for i, t in enumerate(TARGETS):
        print(f"{t:12s} " + " ".join(f"{v[i]:17.3f}" for v in res.values()))
    print(f"{'MEAN':12s} " + " ".join(f"{v.mean():17.3f}" for v in res.values()))
    print(f"null mean R^2 {null.mean():.3f}, 95th pct {np.percentile(null, 95):.3f}; eeg_ridge p = {p_val:.3g}")
    for k, v in out["per_subject_eeg_ridge"].items():
        print(f"  {k}: eeg_ridge mean R^2 {v:.3f}")


if __name__ == "__main__":
    main()
