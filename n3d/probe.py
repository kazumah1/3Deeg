"""How much category information is linearly decodable from the EEG, and do encoder seeds add up?

linear     Multinomial logistic regression on time-binned EEG (no deep model). If this matches
           the deep encoder's ~5% top-1, the ceiling is the data, not the architecture.
           Features: per-channel means in fixed time bins of both streams, z-scored per
           participant on that participant's training trials. C is picked on val NLL.
           Test uses repeat-averaged EEG per (participant, object), like the encoder.
timecourse Same linear probe on short sliding windows of one stream: which time windows
           carry category information (e.g. the 100-300 ms visual response in the static stream)?
ensemble   Average the softmax of several encoder runs' test logits (from `n3d.encoder embed`).

    python -m n3d.probe linear --run lin_sub01
    python -m n3d.probe linear --run lin_multi5 --participants sub01 sub02 sub03 sub04 sub05
    python -m n3d.probe timecourse --run tc_multi5 --participants sub01 sub02 sub03 sub04 sub05 --stream stat
    python -m n3d.probe ensemble --run ens_sub01 --encoder-runs enc_sub01 enc_sub01_s1 enc_sub01_s2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .data import load_bundle, make_splits
from .encoder import run_dir


def bin_means(x: np.ndarray, n_bins: int) -> np.ndarray:
    """(T, C, L) -> (T, C * n_bins), mean over equal time bins (tail samples dropped)."""
    T, C, L = x.shape
    w = L // n_bins
    return x[:, :, : w * n_bins].reshape(T, C, n_bins, w).mean(-1).reshape(T, C * n_bins)


def crop(x: np.ndarray, hz: int, ms: tuple[float, float] | None) -> np.ndarray:
    if not ms:
        return x
    a, z = round(ms[0] * hz / 1000), round(ms[1] * hz / 1000)
    return x[:, :, a:z]


def features(b, dyn_bins: int, stat_bins: int, streams: str, dyn_ms=None, stat_ms=None) -> np.ndarray:
    """Bin widths stay those of the full epoch (dyn 600/dyn_bins samples, stat 250/stat_bins) when cropping."""
    parts = []
    if streams in ("both", "dyn"):
        x = crop(b.dyn, 100, dyn_ms)
        parts.append(bin_means(x, max(1, x.shape[2] // (b.dyn.shape[2] // dyn_bins))))
    if streams in ("both", "stat"):
        x = crop(b.stat, 250, stat_ms)
        parts.append(bin_means(x, max(1, x.shape[2] // (b.stat.shape[2] // stat_bins))))
    return np.concatenate(parts, 1).astype(np.float32)


def averaged(X: np.ndarray, b, objs: np.ndarray):
    """One row per (participant, object): mean of that object's repeats."""
    rows, y, sub, obj = [], [], [], []
    for s in np.unique(b.trial_sub):
        for o in objs:
            t = np.flatnonzero((b.trial_sub == s) & (b.trial_obj == o))
            if len(t):
                rows.append(X[t].mean(0)); y.append(b.obj_cat[o]); sub.append(s); obj.append(o)
    return np.stack(rows), np.array(y), np.array(sub), np.array(obj)


def topk(P: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1).mean())


def metrics(P: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    from scipy.stats import binomtest
    n = len(y)
    out = {"n": n, "top1": topk(P, y, 1), "top5": topk(P, y, 5),
           "nll": float(-np.log(np.clip(P[np.arange(n), y], 1e-12, None)).mean())}
    out["p_top1"] = float(binomtest(round(out["top1"] * n), n, 1 / n_classes, alternative="greater").pvalue)
    out["p_top5"] = float(binomtest(round(out["top5"] * n), n, 5 / n_classes, alternative="greater").pvalue)
    return out


def cmd_linear(args):
    from sklearn.linear_model import LogisticRegression
    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    s = make_splits(b, args.split)
    X = features(b, args.dyn_bins, args.stat_bins, args.streams, args.dyn_ms, args.stat_ms)
    tr = np.isin(b.trial_obj, s["train"])
    X = zscore_per_subject(X, b, s["train"])  # per participant, on its own training trials
    Xtr, ytr = X[tr], b.obj_cat[b.trial_obj[tr]]
    if args.train_on == "averaged":
        Xtr, ytr, _, _ = averaged(X, b, s["train"])
    Xva, yva, _, _ = averaged(X, b, s["val"])
    Xte, yte, ste, _ = averaged(X, b, s["test"])
    K = len(b.categories)
    print(f"features {X.shape[1]}  train rows {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")
    results, best = [], None
    for C in args.Cs:
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
        full = lambda Z: _full_proba(clf, Z, K)
        rec = {"C": C, "train": metrics(full(Xtr), ytr, K), "val": metrics(full(Xva), yva, K)}
        results.append(rec)
        print(f"C {C:g}: train top1 {rec['train']['top1']:.3f} | val top1 {rec['val']['top1']:.3f} "
              f"top5 {rec['val']['top5']:.3f} nll {rec['val']['nll']:.3f}")
        if best is None or rec["val"]["nll"] < best[0]["val"]["nll"]:
            best = (rec, clf)
    rec, clf = best
    P = _full_proba(clf, Xte, K)
    test = {"all": metrics(P, yte, K)}
    for sub in np.unique(ste):
        test[args.participants[sub]] = metrics(P[ste == sub], yte[ste == sub], K)
    out = {"chance_top1": 1 / K, "chance_top5": 5 / K, "uniform_nll": float(np.log(K)),
           "selected_C": rec["C"], "sweep": results, "test": test, "args": {k: str(v) for k, v in vars(args).items()}}
    (run_dir(args.run) / "linear_probe.json").write_text(json.dumps(out, indent=1))
    for k, m in test.items():
        print(f"test {k:6s} n {m['n']:4d}  top1 {m['top1']:.3f} (p={m['p_top1']:.2g})  "
              f"top5 {m['top5']:.3f} (p={m['p_top5']:.2g})  nll {m['nll']:.3f}")
    print(f"chance top1 {1 / K:.3f} top5 {5 / K:.3f} uniform nll {np.log(K):.3f}")


def zscore_per_subject(X: np.ndarray, b, train_objs: np.ndarray) -> np.ndarray:
    tr = np.isin(b.trial_obj, train_objs)
    for sub in np.unique(b.trial_sub):
        m = b.trial_sub == sub
        mu, sd = X[m & tr].mean(0), X[m & tr].std(0) + 1e-6
        X[m] = (X[m] - mu) / sd
    return X


def cmd_timecourse(args):
    from sklearn.linear_model import LogisticRegression
    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    s = make_splits(b, args.split)
    raw, hz = (b.stat, 250) if args.stream == "stat" else (b.dyn, 100)
    win, step = round(args.window_ms * hz / 1000), round(args.step_ms * hz / 1000)
    sub_bins = max(1, win // max(1, round(args.subbin_ms * hz / 1000)))
    tr = np.isin(b.trial_obj, s["train"])
    K, rows = len(b.categories), []
    print(f"{args.stream}: {raw.shape[2]} samples @ {hz} Hz, window {args.window_ms} ms, step {args.step_ms} ms, C {args.C}")
    for start in range(0, raw.shape[2] - win + 1, step):
        X = zscore_per_subject(bin_means(raw[:, :, start:start + win], sub_bins).astype(np.float32), b, s["train"])
        clf = LogisticRegression(C=args.C, max_iter=2000).fit(X[tr], b.obj_cat[b.trial_obj[tr]])
        Xte, yte, _, _ = averaged(X, b, s["test"])
        m = metrics(_full_proba(clf, Xte, K), yte, K)
        m |= {"start_ms": 1000 * start / hz, "end_ms": 1000 * (start + win) / hz}
        rows.append(m)
        print(f"{m['start_ms']:6.0f}-{m['end_ms']:6.0f} ms  top1 {m['top1']:.3f}  top5 {m['top5']:.3f} "
              f"(p={m['p_top5']:.2g})  nll {m['nll']:.3f}")
    (run_dir(args.run) / f"timecourse_{args.stream}.json").write_text(json.dumps(
        {"chance_top5": 5 / K, "uniform_nll": float(np.log(K)), "windows": rows, "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    print(f"chance top5 {5 / K:.3f}, uniform nll {np.log(K):.3f}; p-values are per window, uncorrected")


def _full_proba(clf, Z: np.ndarray, K: int) -> np.ndarray:
    P = np.full((len(Z), K), 1e-12)
    P[:, clf.classes_] = clf.predict_proba(Z)
    return P / P.sum(1, keepdims=True)


def cmd_ensemble(args):
    b = load_bundle(args.data_root, ("sub01",), args.geometry)
    probs, objs = [], None
    for r in args.encoder_runs:
        e = np.load(run_dir(r) / "embeddings.npz")
        o = e["test_avg_obj"]
        if objs is None:
            objs = o
        assert np.array_equal(o, objs), f"{r}: test objects differ; re-run `n3d.encoder embed` on the same data"
        lg = e["test_avg_shape_logits"]
        lg = lg - lg.max(1, keepdims=True)
        probs.append(np.exp(lg) / np.exp(lg).sum(1, keepdims=True))
    y, K = b.obj_cat[objs], probs[0].shape[1]
    out = {r: metrics(p, y, K) for r, p in zip(args.encoder_runs, probs)}
    out["ensemble"] = metrics(np.mean(probs, 0), y, K)
    (run_dir(args.run) / "ensemble.json").write_text(json.dumps(out, indent=1))
    for k, m in out.items():
        print(f"{k:16s} top1 {m['top1']:.3f} (p={m['p_top1']:.2g})  top5 {m['top5']:.3f}  nll {m['nll']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("linear", "timecourse", "ensemble"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--data-root", type=Path, default=None)
        p.add_argument("--geometry", default="point_cloud_simple")
    lin = sub.choices["linear"]
    lin.add_argument("--participants", nargs="+", default=["sub01"])
    lin.add_argument("--split", default="ours", choices=list(config.SPLITS))
    lin.add_argument("--streams", default="both", choices=["both", "dyn", "stat"])
    lin.add_argument("--dyn-bins", type=int, default=30, help="6 s at 100 Hz -> 200 ms bins")
    lin.add_argument("--stat-bins", type=int, default=25, help="1 s at 250 Hz -> 40 ms bins")
    lin.add_argument("--dyn-ms", type=float, nargs=2, default=None, help="crop the dynamic epoch, e.g. 0 1000")
    lin.add_argument("--stat-ms", type=float, nargs=2, default=None, help="crop the static epoch, e.g. 240 920")
    lin.add_argument("--train-on", default="trials", choices=["trials", "averaged"])
    lin.add_argument("--Cs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    tc = sub.choices["timecourse"]
    tc.add_argument("--participants", nargs="+", default=["sub01"])
    tc.add_argument("--split", default="ours", choices=list(config.SPLITS))
    tc.add_argument("--stream", default="stat", choices=["stat", "dyn"])
    tc.add_argument("--window-ms", type=float, default=100)
    tc.add_argument("--step-ms", type=float, default=50)
    tc.add_argument("--subbin-ms", type=float, default=20, help="feature resolution inside a window")
    tc.add_argument("--C", type=float, default=1e-3)
    sub.choices["ensemble"].add_argument("--encoder-runs", nargs="+", required=True)
    args = ap.parse_args(argv)
    {"linear": cmd_linear, "timecourse": cmd_timecourse, "ensemble": cmd_ensemble}[args.cmd](args)


if __name__ == "__main__":
    main()
