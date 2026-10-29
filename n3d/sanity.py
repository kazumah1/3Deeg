"""Model-free checks on the EEG itself: is the weak decoding a data problem?

evoked       Grand-average evoked response per stream: global field power (GFP, spatial std of
             the trial-averaged signal) over time. A clean visual response shows GFP peaks near
             100-200 ms after onset. No early peak points to an epoch / trigger alignment issue.
reliability  Noise ceiling without any training. Train objects have 2 repetitions: correlate the
             time-binned pattern of repetition A with repetition B across objects.
               same_minus_diff_r     mean r(same object) - mean r(different objects)
               rep_identification    is repetition B of an object the best match for A? (N-way)
               same_cat_minus_diff_r same-category different-object r minus different-category r
             Also time-resolved (per window). If same-object > different-object at chance level,
             no model can decode instances; if same-category > different-category, there is
             category structure for a model to find.
lowlevel     Linear probe for the object's colour class (6 classes, the encoder's colour head).
             Colour is a strong low-level visual feature; failing to decode it while category
             decodes weakly would suggest a trial-to-label alignment problem.

    python -m n3d.sanity --run sanity_s1_10 --participants sub01 sub02 ... sub10
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


def gfp_curve(x: np.ndarray) -> np.ndarray:
    """x (T, C, L) -> GFP (L,) of the grand-average evoked response."""
    return x.mean(0).std(0)


def evoked(b, participants) -> dict:
    out = {}
    for name, arr, hz in (("stat", b.stat, 250), ("dyn", b.dyn, 100)):
        per = {}
        for s in np.unique(b.trial_sub):
            g = gfp_curve(arr[b.trial_sub == s])
            per[participants[s]] = g
        grand = np.mean(list(per.values()), 0)
        t = np.arange(len(grand)) * 1000 / hz
        lim = t <= (1000 if name == "stat" else 2000)
        peaks = sorted(np.argsort(-grand[lim])[:3].tolist())
        out[name] = {"ms": t.tolist(), "grand_gfp": grand.tolist(),
                     "per_subject_gfp": {k: v.tolist() for k, v in per.items()},
                     "top3_peak_ms": [float(t[i]) for i in peaks],
                     "early_100_200_vs_baseline_0_50": float(grand[(t >= 100) & (t <= 200)].mean() / grand[t <= 50].mean())}
        print(f"{name}: GFP top peaks at {out[name]['top3_peak_ms']} ms; "
              f"mean GFP 100-200 ms / 0-50 ms = {out[name]['early_100_200_vs_baseline_0_50']:.2f}")
    return out


def rep_pairs(X: np.ndarray, b, objs: np.ndarray):
    """For each (subject, object) with >=2 repetitions: pattern of rep 0 and mean of the rest."""
    A, B, cats, subs = [], [], [], []
    for s in np.unique(b.trial_sub):
        for o in objs:
            t = np.flatnonzero((b.trial_sub == s) & (b.trial_obj == o))
            if len(t) >= 2:
                h = len(t) // 2
                A.append(X[t[:h]].mean(0)); B.append(X[t[h:]].mean(0)); cats.append(b.obj_cat[o]); subs.append(s)
    return np.stack(A), np.stack(B), np.array(cats), np.array(subs)


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-9)
    return A @ B.T / A.shape[1]


def reliability_stats(A, B, cats, subs, rng, n_perm=200) -> dict:
    """Within each subject: rep-A vs rep-B correlations. Permutation null shuffles B's objects."""
    def stats(Bp_index):
        same, diff, same_cat, diff_cat, ident = [], [], [], [], []
        for s in np.unique(subs):
            i = np.flatnonzero(subs == s)
            R = corr_matrix(A[i], B[i][Bp_index[s]])
            n = len(i)
            eye = np.eye(n, dtype=bool)
            c = cats[i]
            sc = (c[:, None] == c[None]) & ~eye
            same.append(R[eye].mean()); diff.append(R[~eye].mean())
            same_cat.append(R[sc].mean()); diff_cat.append(R[(c[:, None] != c[None])].mean())
            ident.append((R.argmax(1) == np.arange(n)).mean())
        return {"same_minus_diff_r": float(np.mean(same) - np.mean(diff)),
                "same_cat_minus_diff_r": float(np.mean(same_cat) - np.mean(diff_cat)),
                "rep_identification": float(np.mean(ident))}
    ident_index = {s: np.arange((subs == s).sum()) for s in np.unique(subs)}
    obs = stats(ident_index)
    null = [stats({s: rng.permutation((subs == s).sum()) for s in np.unique(subs)}) for _ in range(n_perm)]
    for k in list(obs):
        nv = np.array([n[k] for n in null])
        obs[k + "_null_p95"] = float(np.percentile(nv, 95))
        obs[k + "_p"] = float((1 + (nv >= obs[k]).sum()) / (1 + len(nv)))
    obs["identification_chance"] = float(np.mean([1 / (subs == s).sum() for s in np.unique(subs)]))
    return obs


def reliability(b, s, rng, n_perm) -> dict:
    out = {}
    X = np.concatenate([bin_means(b.dyn, 30), bin_means(b.stat, 25)], 1).astype(np.float32)
    X = zscore_per_subject(X, b, s["train"])
    A, B, cats, subs = rep_pairs(X, b, s["train"])
    out["full_epoch"] = reliability_stats(A, B, cats, subs, rng, n_perm)
    r = out["full_epoch"]
    print(f"reliability (train objects, rep A vs rep B, within subject, n_pairs {len(A)}):")
    print(f"  same - diff object r      {r['same_minus_diff_r']:+.4f} (null p95 {r['same_minus_diff_r_null_p95']:+.4f}, p={r['same_minus_diff_r_p']:.3g})")
    print(f"  same-cat - diff-cat r     {r['same_cat_minus_diff_r']:+.4f} (null p95 {r['same_cat_minus_diff_r_null_p95']:+.4f}, p={r['same_cat_minus_diff_r_p']:.3g})")
    print(f"  rep identification        {r['rep_identification']:.4f} (chance {r['identification_chance']:.4f}, p={r['rep_identification_p']:.3g})")
    windows = []
    for name, arr, hz, win, step in (("stat", b.stat, 250, 25, 25), ("dyn", b.dyn, 100, 50, 50)):
        for a in range(0, arr.shape[2] - win + 1, step):
            Xw = zscore_per_subject(bin_means(arr[:, :, a:a + win], 5).astype(np.float32), b, s["train"])
            Aw, Bw, _, _ = rep_pairs(Xw, b, s["train"])
            st = reliability_stats(Aw, Bw, cats, subs, rng, n_perm=50)
            st |= {"stream": name, "start_ms": 1000 * a / hz, "end_ms": 1000 * (a + win) / hz}
            windows.append(st)
            print(f"  {name} {st['start_ms']:5.0f}-{st['end_ms']:5.0f} ms  same-diff {st['same_minus_diff_r']:+.4f} "
                  f"(p={st['same_minus_diff_r_p']:.2g})  cat {st['same_cat_minus_diff_r']:+.4f} (p={st['same_cat_minus_diff_r_p']:.2g})")
    out["windows"] = windows
    return out


def lowlevel(b, s, participants, Cs) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    X = np.concatenate([bin_means(b.dyn, 30), bin_means(b.stat, 25)], 1).astype(np.float32)
    X = zscore_per_subject(X, b, s["train"])
    tr = np.isin(b.trial_obj, s["train"])
    Xtr, ytr = X[tr], b.color[b.trial_obj[tr]]
    Xva, _, _, ova = averaged(X, b, s["val"])
    Xte, _, ste, ote = averaged(X, b, s["test"])
    yva, yte = b.color[ova], b.color[ote]
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        bal = balanced_accuracy_score(yva, clf.predict(Xva))
        if best is None or bal > best[0]:
            best = (bal, C, clf)
    _, C, clf = best
    pred = clf.predict(Xte)
    out = {"C": C, "n_classes": int(len(np.unique(b.color))), "balanced_acc_test": float(balanced_accuracy_score(yte, pred)),
           "chance_balanced": 1 / len(np.unique(b.color)),
           "per_subject": {participants[u]: float(balanced_accuracy_score(yte[ste == u], pred[ste == u])) for u in np.unique(ste)}}
    rng = np.random.default_rng(0)
    null = [balanced_accuracy_score(rng.permutation(yte), pred) for _ in range(1000)]
    out["p"] = float((1 + (np.array(null) >= out["balanced_acc_test"]).sum()) / 1001)
    print(f"colour (6-way, balanced acc): test {out['balanced_acc_test']:.3f} (chance {out['chance_balanced']:.3f}, p={out['p']:.3g}), C {C:g}")
    for k, v in out["per_subject"].items():
        print(f"  {k}: {v:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--geometry", default="point_cloud_simple")
    ap.add_argument("--split", default="ours", choices=list(config.SPLITS))
    ap.add_argument("--checks", nargs="+", default=["evoked", "reliability", "lowlevel"],
                    choices=["evoked", "reliability", "lowlevel"])
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--Cs", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    args = ap.parse_args(argv)
    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    s = make_splits(b, args.split)
    rng = np.random.default_rng(0)
    res = {}
    if "evoked" in args.checks:
        res["evoked"] = evoked(b, args.participants)
    if "reliability" in args.checks:
        res["reliability"] = reliability(b, s, rng, args.permutations)
    if "lowlevel" in args.checks:
        res["lowlevel"] = lowlevel(b, s, args.participants, args.Cs)
    (run_dir(args.run) / "sanity.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {run_dir(args.run) / 'sanity.json'}")


if __name__ == "__main__":
    main()
