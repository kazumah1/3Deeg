"""What does the EEG know about the object? Linear probes with permutation nulls, honest protocol.

Ridge (closed form via one SVD, so permutations are cheap) from binned EEG to several targets:
  cat     72-way category (one-hot ridge)            top1, top5
  color   6-way colour label                         balanced accuracy
  geom    bbox extents x/y/z, log elongation, Rg     Pearson r per target (mean reported)
  clip    1024-d CLIP video embedding                pairwise 2AFC on test objects
alpha is picked on val (variant 07), the model is fit on train (00-06), reported on test (08-09).
Null: the same fit with train objects' targets permuted (object level), n_perm times.

Modes
  subject  per participant: train on single trials, test on that participant's repeat-averaged EEG
  grand    average each object's EEG over all trials of all participants (features z-scored per
           participant first): 5x more trials per object, the most SNR this dataset allows

    python -m n3d.attr_probe attrs  --participants sub01 sub02 sub03 sub04 sub05
    python -m n3d.attr_probe time   --participants sub01 sub02 sub03 sub04 sub05 --fig figs/probe_time.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .data import load_bundle, make_splits

ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
GEOM_NAMES = ("ext_x", "ext_y", "ext_z", "log_elong", "rg")


# ---------------------------------------------------------------- features and targets
def binned(x: np.ndarray, bins: int) -> np.ndarray:
    n, c, t = x.shape
    t2 = t - t % bins
    return x[:, :, :t2].reshape(n, c, bins, t2 // bins).mean(-1).reshape(n, c * bins)


def full_features(b) -> np.ndarray:
    return np.concatenate([binned(b.dyn, 20), binned(b.stat, 10)], 1).astype(np.float32)


def window_features(b, stream: str, start: int, width: int, sub_bins: int = 5) -> np.ndarray:
    x = (b.dyn if stream == "dyn" else b.stat)[:, :, start:start + width]
    return binned(x, sub_bins).astype(np.float32)


def targets(b) -> dict:
    pts = b.points
    ext = pts.max(1) - pts.min(1)
    rg = np.sqrt(((pts - pts.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))
    geom = np.column_stack([ext, np.log(ext.max(1) / np.maximum(ext.min(1), 1e-6)), rg])
    from .encoder import shape_descriptor
    rng = np.random.default_rng(0)
    d2 = np.stack([shape_descriptor(p, rng) for p in pts])  # rotation-invariant 3D geometry, no appearance
    return {"d2": (d2 - d2.mean(0)) / (d2.std(0) + 1e-9), "cat": np.eye(72)[b.obj_cat], "color": np.eye(6)[b.color], "geom": geom, "clip": b.clip_video,
            "_cat": b.obj_cat, "_color": b.color}


# ---------------------------------------------------------------- datasets per mode
def _zscore_per_sub(F, b, train_objs):
    F = F.copy()
    for s in np.unique(b.trial_sub):
        m = b.trial_sub == s
        ref = m & np.isin(b.trial_obj, train_objs)
        mu, sd = F[ref].mean(0), F[ref].std(0) + 1e-6
        F[m] = (F[m] - mu) / sd
    return F


def _avg(F, trial_obj, objs, mask=None):
    m = np.ones(len(F), bool) if mask is None else mask
    return np.stack([F[m & (trial_obj == o)].mean(0) for o in objs])


def datasets(F, b, splits, mode: str):
    """Yield (label, Xtr, otr, Xva, Xte) with object ids otr for train rows."""
    F = _zscore_per_sub(F, b, splits["train"])
    if mode == "grand":
        yield "grand", _avg(F, b.trial_obj, splits["train"]), splits["train"], \
            _avg(F, b.trial_obj, splits["val"]), _avg(F, b.trial_obj, splits["test"])
        return
    for s in np.unique(b.trial_sub):
        m = b.trial_sub == s
        tr = m & np.isin(b.trial_obj, splits["train"])
        yield f"sub{s}", F[tr], b.trial_obj[tr], _avg(F, b.trial_obj, splits["val"], m), \
            _avg(F, b.trial_obj, splits["test"], m)


# ---------------------------------------------------------------- ridge + metrics
class Ridge:
    def __init__(self, X):
        self.mu = X.mean(0)
        self.U, self.s, self.Vt = np.linalg.svd(X - self.mu, full_matrices=False)

    def predict(self, Y, Xs, alpha):
        ym = Y.mean(0)
        W = self.Vt.T @ ((self.s / (self.s ** 2 + alpha))[:, None] * (self.U.T @ (Y - ym)))
        return [(X - self.mu) @ W + ym for X in Xs]


def _pearson_cols(P, Y):
    P, Y = P - P.mean(0), Y - Y.mean(0)
    return (P * Y).sum(0) / np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0) + 1e-12)


def pair_2afc(P, Y, cat=None, color=None):
    Pz = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
    Yz = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-9)
    C = Pz @ Yz.T / P.shape[1]
    d = np.diag(C)
    good = (d[:, None] + d[None, :]) > (C + C.T)
    iu = np.triu_indices(len(C), 1)
    out = {"2afc": float(good[iu].mean())}
    if cat is not None:  # same-category pairs ask for within-category shape/appearance, the rest mostly category
        same = (cat[:, None] == cat[None, :])[iu]
        out |= {"2afc_within": float(good[iu][same].mean()), "2afc_between": float(good[iu][~same].mean())}
        if color is not None:  # colour control: does within-category success need the siblings to differ in colour?
            sc = (color[:, None] == color[None, :])[iu]
            out |= {"2afc_within_samecol": float(good[iu][same & sc].mean()),
                    "2afc_within_diffcol": float(good[iu][same & ~sc].mean())}
    return out


def metric(name, P, obj, T):
    if name == "cat":
        y = T["_cat"][obj]
        rank = np.argsort(-P, 1)
        return {"top1": float((rank[:, 0] == y).mean()), "top5": float((rank[:, :5] == y[:, None]).any(1).mean())}
    if name == "color":
        y, yh = T["_color"][obj], P.argmax(1)
        return {"bal_acc": float(np.mean([(yh[y == c] == c).mean() for c in np.unique(y)]))}
    if name == "geom":
        r = _pearson_cols(P, T["geom"][obj])
        return {"r_mean": float(r.mean()), **{f"r_{n}": float(v) for n, v in zip(GEOM_NAMES, r)}}
    return pair_2afc(P, T[name][obj], T["_cat"][obj], T["_color"][obj])


PRIMARY = {"cat": "top5", "color": "bal_acc", "geom": "r_mean", "clip": "2afc", "d2": "2afc"}


def probe(Xtr, otr, Xva, Xte, splits, T, names, n_perm, rng) -> dict:
    model, out = Ridge(Xtr), {}
    for nm in names:
        Yobj = T[nm]
        key = PRIMARY[nm]
        va = {a: metric(nm, model.predict(Yobj[otr], [Xva], a)[0], splits["val"], T)[key] for a in ALPHAS}
        a = max(va, key=va.get)
        res = metric(nm, model.predict(Yobj[otr], [Xte], a)[0], splits["test"], T)
        tr_objs = splits["train"]
        null = []
        for _ in range(n_perm):
            perm = dict(zip(tr_objs.tolist(), rng.permutation(tr_objs).tolist()))
            Yp = Yobj[[perm[int(o)] for o in otr]]
            null.append(metric(nm, model.predict(Yp, [Xte], a)[0], splits["test"], T))
        if not null:
            out[nm] = res | {"alpha": a, "val": va[a]}
            continue
        extra = {}
        for k in res:
            nk = np.array([n[k] for n in null])
            extra |= {f"null_mean_{k}": float(nk.mean()), f"null_p95_{k}": float(np.percentile(nk, 95)),
                      f"p_{k}": float(((nk >= res[k]).sum() + 1) / (len(nk) + 1))}
        out[nm] = res | extra | {"alpha": a, "val": va[a], "null_mean": extra[f"null_mean_{key}"],
                                 "null_p95": extra[f"null_p95_{key}"], "p_perm": extra[f"p_{key}"]}
    return out


def _mean_over(rows: list[dict]) -> dict:
    out = {}
    for nm in rows[0]:
        out[nm] = {k: float(np.mean([r[nm][k] for r in rows])) for k in rows[0][nm]}
    return out


# ---------------------------------------------------------------- commands
def load(participants):
    b = load_bundle(None, tuple(participants))
    return b, make_splits(b, "ours"), targets(b)


def attrs(a):
    b, splits, T = load(a.participants)
    F, rng, res = full_features(b), np.random.default_rng(0), {}
    for mode in ("subject", "grand"):
        rows = {lab: probe(Xtr, otr, Xva, Xte, splits, T, ("cat", "color", "geom", "clip"), a.n_perm, rng)
                for lab, Xtr, otr, Xva, Xte in datasets(F, b, splits, mode)}
        res[mode] = {"per": rows, "mean": _mean_over(list(rows.values()))}
    lines = ["| mode | target | metric | test | null mean | null 95% | p (perm) |", "|---|---|---|---|---|---|---|"]
    for mode in res:
        for nm, r in res[mode]["mean"].items():
            k = PRIMARY[nm]
            lines.append(f"| {mode} | {nm} | {k} | {r[k]:.4f} | {r['null_mean']:.4f} | {r['null_p95']:.4f} | {r['p_perm']:.3f} |")
    for mode in res:
        c = res[mode]["mean"]["clip"]
        lines.append(f"\n{mode} clip 2afc within-cat {c['2afc_within']:.4f} (null95 {c['null_p95_2afc_within']:.4f}, "
                     f"p {c['p_2afc_within']:.3f}) | between-cat {c['2afc_between']:.4f} "
                     f"(null95 {c['null_p95_2afc_between']:.4f}, p {c['p_2afc_between']:.3f})")
    geo = res["grand"]["mean"]["geom"]
    lines.append("\ngrand geom r: " + ", ".join(f"{n} {geo['r_' + n]:.3f}" for n in GEOM_NAMES))
    md = "\n".join(lines)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "probe_attrs.json").write_text(json.dumps(res, indent=1))
    (out / "probe_attrs.md").write_text(md + "\n")
    print(md)


def timecourse(a):
    b, splits, T = load(a.participants)
    rng, res = np.random.default_rng(0), {}
    wins = {"dyn": [(s, 50) for s in range(0, config.DYN_LEN - 50 + 1, 25)],      # 500 ms @100 Hz
            "stat": [(s, 50) for s in range(0, config.STAT_LEN - 50 + 1, 25)]}    # 200 ms @250 Hz
    hz = {"dyn": 100, "stat": 250}
    for stream, ws in wins.items():
        for mode in ("subject", "grand"):
            series = []
            for s, w in ws:
                F = window_features(b, stream, s, w)
                rows = [probe(Xtr, otr, Xva, Xte, splits, T, ("clip", "color"), a.n_perm, rng)
                        for _, Xtr, otr, Xva, Xte in datasets(F, b, splits, mode)]
                m = _mean_over(rows)
                series.append({"t_ms": 1000 * (s + w / 2) / hz[stream], **{f"{nm}_{k}": v for nm, r in m.items()
                                                                         for k, v in r.items()}})
                print(stream, mode, f"{series[-1]['t_ms']:.0f}ms clip2afc {m['clip']['2afc']:.3f} "
                      f"(null95 {m['clip']['null_p95']:.3f})  color {m['color']['bal_acc']:.3f}", flush=True)
            res[f"{stream}_{mode}"] = series
            best = max(series, key=lambda r: r["clip_val"])  # window chosen on val, reported on test
            print(f"{stream} {mode} val-picked window {best['t_ms']:.0f}ms: test clip2afc {best['clip_2afc']:.3f} "
                  f"(within {best['clip_2afc_within']:.3f}, null95 {best['clip_null_p95']:.3f})", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "probe_time.json").write_text(json.dumps(res, indent=1))
    if a.fig:
        plot(res, Path(a.fig))


def window(a):
    """One fixed window (chosen beforehand, e.g. on val by `time`), all targets, many permutations."""
    b, splits, T = load(a.participants)
    F, rng, res = window_features(b, a.stream, a.start, a.width), np.random.default_rng(1), {}
    for mode in ("subject", "grand"):
        rows = [probe(Xtr, otr, Xva, Xte, splits, T, ("clip", "d2", "geom", "cat"), a.n_perm, rng)
                for _, Xtr, otr, Xva, Xte in datasets(F, b, splits, mode)]
        res[mode] = m = _mean_over(rows)
        c = m["clip"]
        print(f"{mode}: clip 2afc {c['2afc']:.3f} (p {c['p_2afc']:.4f}) | within {c['2afc_within']:.3f} "
              f"(null mean {c['null_mean_2afc_within']:.3f}, 95% {c['null_p95_2afc_within']:.3f}, p {c['p_2afc_within']:.4f}) "
              f"| between {c['2afc_between']:.3f} (p {c['p_2afc_between']:.4f}) | geom r {m['geom']['r_mean']:.3f} "
              f"(p {m['geom']['p_r_mean']:.4f}) | cat top5 {m['cat']['top5']:.3f} (p {m['cat']['p_top5']:.4f})")
        print(f"   colour control: within same-colour {c['2afc_within_samecol']:.3f} (p {c['p_2afc_within_samecol']:.4f}) "
              f"| diff-colour {c['2afc_within_diffcol']:.3f} (p {c['p_2afc_within_diffcol']:.4f})")
        d = m["d2"]
        print(f"   D2 shape target: all {d['2afc']:.3f} (p {d['p_2afc']:.4f}) | within {d['2afc_within']:.3f} "
              f"(p {d['p_2afc_within']:.4f}) | between {d['2afc_between']:.3f} (p {d['p_2afc_between']:.4f})")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"probe_window_{a.stream}_{a.start}_{a.width}.json").write_text(json.dumps(res, indent=1))


def export(a):
    """Predicted CLIP vectors for test objects (ridge, alpha on val) -> npz for diffusion sample --clip-file.
    norm=match rescales each dimension of the predictions (label-free, from the test predictions themselves)
    to the training CLIP mean/sd, undoing ridge shrinkage. --perm-seed fits on permuted targets (null)."""
    b, splits, T = load(a.participants)
    F = full_features(b) if a.stream == "full" else window_features(b, a.stream, a.start, a.width)
    Y = T["clip"]
    for lab, Xtr, otr, Xva, Xte in datasets(F, b, splits, a.mode):
        Ytr = Y[otr]
        if a.perm_seed is not None:
            tr = splits["train"]
            perm = dict(zip(tr.tolist(), np.random.default_rng(a.perm_seed).permutation(tr).tolist()))
            Ytr = Y[[perm[int(o)] for o in otr]]
        model = Ridge(Xtr)
        va = {al: pair_2afc(model.predict(Ytr, [Xva], al)[0], Y[splits["val"]])["2afc"] for al in ALPHAS}
        al = max(va, key=va.get)
        P = model.predict(Ytr, [Xte], al)[0]
        mu_c, sd_c = Y[splits["train"]].mean(0), Y[splits["train"]].std(0)
        Pm = (P - P.mean(0)) / (P.std(0) + 1e-9) * sd_c + mu_c
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        name = f"clippred_{a.stream}_{lab}" + (f"_perm{a.perm_seed}" if a.perm_seed is not None else "")
        # on-manifold projections: the decoder only saw real CLIP vectors, so snap predictions onto training CLIP
        Ctr = Y[splits["train"]]
        Pz = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
        Cz = (Ctr - Ctr.mean(1, keepdims=True)) / (Ctr.std(1, keepdims=True) + 1e-9)
        nn = np.argsort(-(Pz @ Cz.T), 1)
        P1, P5 = Ctr[nn[:, 0]], Ctr[nn[:, :5]].mean(1)
        for norm, arr in (("none", P), ("match", Pm), ("nn1", P1), ("nn5", P5)):
            np.savez_compressed(out / f"{name}_{norm}.npz", obj=splits["test"], clip=arr.astype(np.float32))
        print(f"{name}: alpha {al:g} val 2afc {va[al]:.3f} test 2afc {pair_2afc(P, Y[splits['test']])['2afc']:.3f}")


def levels(a):
    """How coarse is the decoded information? Cluster the 72 training-CLIP category centroids into k groups
    (k-means, k = 2..72); decode category by nearest centroid of the ridge-predicted CLIP, score at each level."""
    from sklearn.cluster import KMeans
    b, splits, T = load(a.participants)
    F = full_features(b) if a.stream == "full" else window_features(b, a.stream, a.start, a.width)
    Y, cat = T["clip"], T["_cat"]
    tr, te = splits["train"], splits["test"]
    M = np.stack([Y[tr][cat[tr] == c].mean(0) for c in range(72)])
    Mz = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    ks = (2, 3, 4, 6, 8, 12, 18, 24, 36, 72)
    lab = {k: (np.arange(72) if k == 72 else KMeans(k, n_init=10, random_state=0).fit_predict(M)) for k in ks}
    rng, res = np.random.default_rng(2), {}

    def acc(P, k):
        Pz = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
        return float((lab[k][(Pz @ Mz.T).argmax(1)] == lab[k][cat[te]]).mean())

    for mode in ("subject", "grand"):
        rows = []
        for _, Xtr, otr, Xva, Xte in datasets(F, b, splits, mode):
            model = Ridge(Xtr)
            va = {al: pair_2afc(model.predict(Y[otr], [Xva], al)[0], Y[splits["val"]])["2afc"] for al in ALPHAS}
            al = max(va, key=va.get)
            P = model.predict(Y[otr], [Xte], al)[0]
            null = []
            for _ in range(a.n_perm):
                perm = dict(zip(tr.tolist(), rng.permutation(tr).tolist()))
                Pn = model.predict(Y[[perm[int(o)] for o in otr]], [Xte], al)[0]
                null.append([acc(Pn, k) for k in ks])
            null = np.array(null)
            rows.append({k: {"acc": acc(P, k), "null_mean": float(null[:, i].mean()),
                             "null_p95": float(np.percentile(null[:, i], 95))} for i, k in enumerate(ks)})
        res[mode] = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in ("acc", "null_mean", "null_p95")} for k in ks}
        for k in ks:
            r = res[mode][k]
            print(f"{mode:8s} k={k:2d}  acc {r['acc']:.3f}  null mean {r['null_mean']:.3f}  null95 {r['null_p95']:.3f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"probe_levels_{a.stream}.json").write_text(json.dumps({"ks": ks, "res": res,
        "groups": {k: lab[k].tolist() for k in ks}, "categories": b.categories}, indent=1))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for mode, c in (("subject", "tab:blue"), ("grand", "tab:orange")):
            ax.plot(ks, [res[mode][k]["acc"] for k in ks], "o-", c=c, label=f"EEG ({mode})")
            ax.fill_between(ks, [res[mode][k]["null_mean"] for k in ks], [res[mode][k]["null_p95"] for k in ks], color=c, alpha=0.15)
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("number of CLIP super-classes (k-means on category centroids)"); ax.set_ylabel("test accuracy")
        ax.set_title("EEG decodes coarse groups, not fine categories; shaded = permutation null", fontsize=8)
        ax.legend(fontsize=8)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=200, bbox_inches="tight")
        print("wrote", a.fig)


def crossfit(a):
    """Condition table for a noise-aware decoder: ridge-decoded CLIP for every object, where each training
    variant (00-06) is predicted by a ridge fit on the other six (so the decoder trains on out-of-sample
    predictions with test-like noise), val/test by the fit on all training objects. alpha picked on val.
    Writes <out>/table_<stream>_grand.npz and a null twin whose training rows are shuffled across objects."""
    b, splits, T = load(a.participants)
    F = full_features(b) if a.stream == "full" else window_features(b, a.stream, a.start, a.width)
    Y = T["clip"]
    (_, Xtr, otr, Xva, Xte), = list(datasets(F, b, splits, "grand"))
    model = Ridge(Xtr)
    va = {al: pair_2afc(model.predict(Y[otr], [Xva], al)[0], Y[splits["val"]])["2afc"] for al in ALPHAS}
    al = max(va, key=va.get)
    table = Y.copy()
    Pva, Pte = model.predict(Y[otr], [Xva, Xte], al)
    table[splits["val"]], table[splits["test"]] = Pva, Pte
    var = b.obj_variant[otr]
    for v in np.unique(var):
        m = var == v
        table[otr[m]] = Ridge(Xtr[~m]).predict(Y[otr[~m]], [Xtr[m]], al)[0]
    cf = pair_2afc(table[otr], Y[otr])["2afc"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allobj = np.arange(len(Y))
    np.savez_compressed(out / f"table_{a.stream}_grand.npz", obj=allobj, clip=table.astype(np.float32))
    null = table.copy()
    null[otr] = table[otr][np.random.default_rng(0).permutation(len(otr))]
    np.savez_compressed(out / f"table_{a.stream}_grand_null.npz", obj=allobj, clip=null.astype(np.float32))
    print(f"alpha {al:g}: cross-fitted train 2afc {cf:.3f}, val {va[al]:.3f}, "
          f"test {pair_2afc(Pte, Y[splits['test']])['2afc']:.3f}; wrote tables to {out}")


def scaling(a):
    """Is decoding SNR-limited? Grand-average every subset of n participants (n = 1..5) using r test repeats
    (r = 1, 2, 4), refit the ridge (alpha on val), score test CLIP 2AFC and geometry r. Then fit
    2AFC = Phi(d/sqrt2), d(N) = d_max * sqrt(N / (N + c)), N = n*r trials averaged per test object."""
    from itertools import combinations
    from types import SimpleNamespace
    from scipy.optimize import curve_fit
    from scipy.stats import norm
    b, splits, T = load(a.participants)
    F = full_features(b) if a.stream == "full" else window_features(b, a.stream, a.start, a.width)
    rep = np.zeros(len(b.trial_obj), int)  # repeat index within (participant, object)
    seen = {}
    for t, key in enumerate(zip(b.trial_sub.tolist(), b.trial_obj.tolist())):
        rep[t] = seen.get(key, 0); seen[key] = rep[t] + 1
    is_test = np.isin(b.trial_obj, splits["test"])
    subs = np.unique(b.trial_sub)
    rows = []
    for n in range(1, len(subs) + 1):
        for combo in combinations(subs.tolist(), n):
            for r in (1, 2, 4):
                keep = np.isin(b.trial_sub, combo) & (~is_test | (rep < r))
                bv = SimpleNamespace(trial_sub=b.trial_sub[keep], trial_obj=b.trial_obj[keep])
                (_, Xtr, otr, Xva, Xte), = list(datasets(F[keep], bv, splits, "grand"))
                res = probe(Xtr, otr, Xva, Xte, splits, T, ("clip", "geom"), 0, np.random.default_rng(0))
                rows.append({"n": n, "r": r, "subs": list(combo), "clip_2afc": res["clip"]["2afc"],
                             "clip_within": res["clip"]["2afc_within"], "geom_r": res["geom"]["r_mean"]})
        print(f"n={n} done", flush=True)
    agg = {}
    for n in range(1, len(subs) + 1):
        for r in (1, 2, 4):
            g = [x for x in rows if x["n"] == n and x["r"] == r]
            agg[(n, r)] = {k: (float(np.mean([x[k] for x in g])), float(np.std([x[k] for x in g])))
                           for k in ("clip_2afc", "clip_within", "geom_r")}
    Ns = np.array([n * r for (n, r) in agg], float)
    pc = np.clip([agg[k]["clip_2afc"][0] for k in agg], 0.501, 0.999)
    dp = np.sqrt(2) * norm.ppf(pc)
    f = lambda N, dmax, c: dmax * np.sqrt(N / (N + c))
    (dmax, c), _ = curve_fit(f, Ns, dp, p0=(2.0, 10.0), bounds=([0, 0], [20, 1e4]))
    ceil = float(norm.cdf(dmax / np.sqrt(2)))
    need = {}
    for target in (0.85, 0.9, 0.95):
        d = np.sqrt(2) * norm.ppf(target)
        need[target] = float(c * d ** 2 / (dmax ** 2 - d ** 2)) if d < dmax else None
    lines = ["| participants n | test repeats r | trials N | CLIP 2AFC | within-cat | geom r |", "|---|---|---|---|---|---|"]
    for (n, r), v in agg.items():
        lines.append(f"| {n} | {r} | {n * r} | {v['clip_2afc'][0]:.3f}±{v['clip_2afc'][1]:.3f} | "
                     f"{v['clip_within'][0]:.3f} | {v['geom_r'][0]:.3f}±{v['geom_r'][1]:.3f} |")
    lines.append(f"\nfit d(N) = {dmax:.2f} sqrt(N/(N+{c:.1f})): ceiling 2AFC {ceil:.3f}; trials per object needed: "
                 + ", ".join(f"{t}: {('%.0f' % v) if v else 'unreachable'}" for t, v in need.items()))
    print("\n".join(lines))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"probe_scaling_{a.stream}.json").write_text(json.dumps(
        {"rows": rows, "fit": {"dmax": dmax, "c": c, "ceiling": ceil, "need": need}}, indent=1))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for r, mk in ((1, "o"), (2, "s"), (4, "^")):
            ks = [(n, rr) for (n, rr) in agg if rr == r]
            ax.errorbar([n * r for n, _ in ks], [agg[k]["clip_2afc"][0] for k in ks],
                        yerr=[agg[k]["clip_2afc"][1] for k in ks], fmt=mk, label=f"{r} repeat(s) x n participants", capsize=2)
        NN = np.logspace(0, np.log10(200), 100)
        ax.plot(NN, norm.cdf(f(NN, dmax, c) / np.sqrt(2)), "k--", lw=1, label=f"fit, ceiling {ceil:.2f}")
        ax.axhline(0.5, c="gray", lw=0.5)
        ax.set_xscale("log"); ax.set_xlabel("EEG trials averaged per test object (N = n x r)")
        ax.set_ylabel("CLIP pairwise 2AFC (test)"); ax.legend(fontsize=7)
        ax.set_title("Decoding grows with averaging: SNR-limited", fontsize=9)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=200, bbox_inches="tight")
        print("wrote", a.fig)


def silhouettes(pts: np.ndarray, views: int = 8, res: int = 32) -> np.ndarray:
    """Rotation-averaged-video proxy for the stimulus image: occupancy of the point cloud projected onto the
    image plane (x', y up) at `views` rotations about the vertical axis, blurred, concatenated."""
    from scipy.ndimage import gaussian_filter
    out = np.zeros((len(pts), views, res, res), np.float32)
    for v, th in enumerate(np.linspace(0, 2 * np.pi, views, endpoint=False)):
        x = pts[..., 0] * np.cos(th) + pts[..., 2] * np.sin(th)
        y = pts[..., 1]
        for i in range(len(pts)):
            h, _, _ = np.histogram2d(y[i], x[i], bins=res, range=[[-1, 1], [-1, 1]])
            out[i, v] = gaussian_filter((h > 0).astype(np.float32), 1.0)
    return out.reshape(len(pts), -1)


def _corr_rdm(X):
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    return 1 - Z @ Z.T / X.shape[1]


def _split_half_rdm(A, B):
    Az = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    Bz = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-9)
    C = Az @ Bz.T / A.shape[1]
    return 1 - (C + C.T) / 2, float(np.mean(np.diag(C)) - np.mean(C[~np.eye(len(C), dtype=bool)]))


def _rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def _partial(y, x, others):
    if not others:
        return float(np.corrcoef(y, x)[0, 1])
    Z = np.column_stack([np.ones(len(y)), *others])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def rsa_models(b, objs):
    from .encoder import shape_descriptor
    pts = b.points[objs]
    rng = np.random.default_rng(0)
    d2 = np.stack([shape_descriptor(p, rng) for p in pts]); d2 = (d2 - d2.mean(0)) / (d2.std(0) + 1e-9)
    sil = silhouettes(pts[:, rng.choice(pts.shape[1], 2048, replace=False)])
    cat, col = b.obj_cat[objs], b.color[objs]
    return {"category": (cat[:, None] != cat[None, :]).astype(float),
            "colour": (col[:, None] != col[None, :]).astype(float),
            "silhouette": _corr_rdm(sil),
            "clip": _corr_rdm(b.clip_video[objs]),
            "geometry_d2": np.sqrt(((d2[:, None] - d2[None]) ** 2).sum(-1))}


def rsa_eeg(F, b, splits, objs):
    """Cross-validated EEG RDM over objs: repeats {0,1} vs {2,3}, each averaged over all participants."""
    F = _zscore_per_sub(F, b, splits["train"])
    rep = np.zeros(len(b.trial_obj), int); seen = {}
    for t, key in enumerate(zip(b.trial_sub.tolist(), b.trial_obj.tolist())):
        rep[t] = seen.get(key, 0); seen[key] = rep[t] + 1
    A = _avg(F, b.trial_obj, objs, rep < 2)
    B = _avg(F, b.trial_obj, objs, rep >= 2)
    return _split_half_rdm(A, B)


def rsa(a):
    """Which stimulus description does the EEG similarity structure follow? Spearman between the cross-validated
    EEG RDM (144 test objects) and model RDMs, plus partial Spearman (each model controlling for all others),
    with Mantel permutation p-values (object labels permuted). Also per stat-stream window."""
    b, splits, T = load(a.participants)
    objs = splits["test"]
    M = rsa_models(b, objs)
    iu = np.triu_indices(len(objs), 1)
    names = list(M)
    rk = {k: _rank(M[k][iu]) for k in names}
    rng = np.random.default_rng(3)
    perms = [rng.permutation(len(objs)) for _ in range(a.n_perm)]

    def score(D):
        y = _rank(D[iu])
        out = {}
        for k in names:
            others = [rk[o] for o in names if o != k]
            r, pr = float(np.corrcoef(y, rk[k])[0, 1]), _partial(y, rk[k], others)
            nr, npr = [], []
            for pm in perms:
                yp = _rank(D[np.ix_(pm, pm)][iu])
                nr.append(np.corrcoef(yp, rk[k])[0, 1]); npr.append(_partial(yp, rk[k], others))
            out[k] = {"rho": r, "p": float((np.sum(np.array(nr) >= r) + 1) / (len(nr) + 1)),
                      "partial": pr, "p_partial": float((np.sum(np.array(npr) >= pr) + 1) / (len(npr) + 1))}
        return out

    res = {"model_intercorr": {f"{x}~{y}": float(np.corrcoef(rk[x], rk[y])[0, 1])
                               for i, x in enumerate(names) for y in names[i + 1:]}}
    windows = [("full", None)] + [("stat", (s, 50)) for s in range(0, config.STAT_LEN - 50 + 1, 25)]
    res["windows"] = []
    for stream, w in windows:
        F = full_features(b) if stream == "full" else window_features(b, stream, *w)
        D, rel = rsa_eeg(F, b, splits, objs)
        sc = score(D)
        t = "full" if w is None else f"{1000 * (w[0] + w[1] / 2) / 250:.0f}ms"
        res["windows"].append({"window": t, "reliability": rel, **sc})
        print(f"{t:6s} rel {rel:.3f} | " + " | ".join(
            f"{k} {v['rho']:.3f} (p {v['p']:.3f}) partial {v['partial']:.3f} (p {v['p_partial']:.3f})" for k, v in sc.items()), flush=True)
    print("model inter-correlations:", {k: round(v, 2) for k, v in res["model_intercorr"].items()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "probe_rsa.json").write_text(json.dumps(res, indent=1))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ws = [r for r in res["windows"] if r["window"] != "full"]
        t = [float(r["window"][:-2]) for r in ws]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        for ax, key, title in ((axes[0], "rho", "Spearman with EEG RDM"), (axes[1], "partial", "partial (controls for all other models)")):
            for k in names:
                ax.plot(t, [r[k][key] for r in ws], "o-", label=k, ms=3)
                sig = [(tt, r[k][key]) for tt, r in zip(t, ws) if r[k]["p" if key == "rho" else "p_partial"] < 0.05]
                if sig:
                    ax.scatter(*zip(*sig), s=40, facecolors="none", edgecolors="k", lw=0.8)
            ax.axhline(0, c="gray", lw=0.5); ax.set_xlabel("static-image window centre (ms)"); ax.set_title(title, fontsize=9)
        axes[0].legend(fontsize=7); axes[0].set_ylabel("rho (144 test objects)")
        fig.suptitle("RSA: which description of the object does the EEG similarity follow? circles: Mantel p < .05 (uncorrected)", fontsize=9)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, dpi=200, bbox_inches="tight")
        print("wrote", a.fig)


def plot(res, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row", gridspec_kw={"hspace": 0.45})
    for j, stream in enumerate(("dyn", "stat")):
        for i, (key, lab, ch) in enumerate((("clip_2afc", "CLIP pairwise 2AFC", 0.5), ("color_bal_acc", "colour bal. acc.", 1 / 6))):
            ax = axes[i, j]
            for mode, c in (("subject", "tab:blue"), ("grand", "tab:orange")):
                s = res.get(f"{stream}_{mode}", [])
                if not s:
                    continue
                t = [r["t_ms"] for r in s]
                ax.plot(t, [r[key] for r in s], c=c, label=mode)
                pre = key.split("_")[0]
                ax.fill_between(t, [r[f"{pre}_null_mean"] for r in s], [r[f"{pre}_null_p95"] for r in s], color=c, alpha=0.15)
            ax.axhline(ch, c="k", lw=0.5, ls="--")
            ax.set_title(f"{'rotating video (6 s)' if stream == 'dyn' else 'static image (1 s)'}: {lab}", fontsize=9)
            ax.set_xlabel("window centre (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Sliding-window linear decoding; shaded = permutation null mean..95th pct", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("attrs", "time", "window", "export", "levels", "crossfit", "scaling", "rsa"):
        p = sub.add_parser(name)
        p.add_argument("--participants", nargs="+", default=["sub01", "sub02", "sub03", "sub04", "sub05"])
        p.add_argument("--n-perm", type=int, default={"attrs": 50, "rsa": 200}.get(name, 20))
        p.add_argument("--out", default=str(config.default_runs_root() / "probe"))
        if name == "time":
            p.add_argument("--fig", default="figs/probe_time.png")
        if name == "rsa":
            p.add_argument("--fig", default="figs/probe_rsa.png")
        if name == "scaling":
            p.add_argument("--stream", default="stat", choices=["stat", "dyn", "full"])
            p.add_argument("--start", type=int, default=75)
            p.add_argument("--width", type=int, default=50)
            p.add_argument("--fig", default="figs/probe_scaling.png")
        if name == "crossfit":
            p.add_argument("--stream", default="stat", choices=["stat", "dyn", "full"])
            p.add_argument("--start", type=int, default=75)
            p.add_argument("--width", type=int, default=50)
        if name == "levels":
            p.add_argument("--stream", default="stat", choices=["stat", "dyn", "full"])
            p.add_argument("--start", type=int, default=75)
            p.add_argument("--width", type=int, default=50)
            p.add_argument("--fig", default="figs/probe_levels.png")
        if name == "export":
            p.add_argument("--stream", default="stat", choices=["stat", "dyn", "full"])
            p.add_argument("--start", type=int, default=75)
            p.add_argument("--width", type=int, default=50)
            p.add_argument("--mode", default="grand", choices=["grand", "subject"])
            p.add_argument("--perm-seed", type=int, default=None)
        if name == "window":
            p.add_argument("--stream", default="stat", choices=["stat", "dyn"])
            p.add_argument("--start", type=int, default=75, help="samples; stat 75 @250 Hz = 300 ms")
            p.add_argument("--width", type=int, default=50)
    a = ap.parse_args(argv)
    {"attrs": attrs, "time": timecourse, "window": window, "export": export, "levels": levels, "crossfit": crossfit, "scaling": scaling, "rsa": rsa}[a.cmd](a)


if __name__ == "__main__":
    main()
