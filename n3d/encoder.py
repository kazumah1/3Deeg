"""Train, evaluate and probe the upstream Neuro-3D EEG encoder.

Training is a faithful port of classification/retri_shape_color.py:
  AdamW(lr 1e-3, wd 5e-4), OneCycle(pct_start 2/epochs, final_div 1e4), batch 128, 200 epochs,
  loss = 0.99*10*MSE + 0.01*10*CLIP-contrastive + 0.1*(shape CE + weighted colour CE),
  both heads aligned to the CLIP *video* feature of the viewed object.
Differences, on purpose:
  * checkpoints are selected on a validation split (variant 07), not on test;
  * training resumes from Drive after a Colab disconnect;
  * every metric is written to metrics.jsonl.

    python -m n3d.encoder train --run enc_sub01
    python -m n3d.encoder train --run enc_tiny --tiny          # overfit check, ~1 min on GPU
    python -m n3d.encoder diagnose --run enc_sub01             # failure-point tests
    python -m n3d.encoder embed --run enc_sub01                # cache f_g / f_a for downstream
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config
from .data import EEG3D, Bundle, load_bundle, make_splits
from .upstream import build_encoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward(model, batch, dev, dyn_scale=1.0, stat_scale=1.0):
    dyn = batch["dyn"].to(dev).float() * dyn_scale
    stat = batch["stat"].to(dev).float() * stat_scale
    f_g, shape_logits, f_a, color_logits = model(dyn, stat)
    return f_g, shape_logits, f_a, color_logits


def loss_fn(model, out, batch, dev, color_w):
    f_g, shape_logits, f_a, color_logits = out
    clip = batch["clip"].to(dev).float()
    scale = model.logit_scale
    contrastive = model.loss_func(f_g, clip, scale) + model.loss_func(f_a, clip, scale)
    regress = F.mse_loss(f_a, clip) + F.mse_loss(f_g, clip)
    cls = F.cross_entropy(color_logits, batch["color"].to(dev), weight=color_w) + \
        F.cross_entropy(shape_logits, batch["cat"].to(dev))
    loss = 0.99 * regress * 10 + 0.01 * contrastive * 10 + cls * 0.1
    return loss, {"regress": regress.item(), "contrastive": contrastive.item(), "cls": cls.item()}


@torch.no_grad()
def embed(model, ds: EEG3D, dev, batch_size=256, **scales) -> dict:
    model.eval()
    keys = ("f_g", "shape_logits", "f_a", "color_logits")  # order returned by the upstream model
    acc = {k: [] for k in keys} | {"obj": [], "cat": [], "color": [], "sub": []}
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        for k, v in zip(keys, forward(model, batch, dev, **scales)):
            acc[k].append(v.float().cpu())
        for k in ("obj", "cat", "color", "sub"):
            acc[k].append(torch.as_tensor(batch[k]))
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def topk(logits: np.ndarray, target: np.ndarray, k: int) -> float:
    return float((np.argsort(-logits, axis=1)[:, :k] == target[:, None]).any(axis=1).mean())


def score(e: dict, b: Bundle, eval_objs: np.ndarray, scale: float = 1.0) -> dict:
    """Classification and retrieval metrics on one split.

    retr_cat_top1 is upstream's 'retrieval accuracy': f_g vs the CLIP video feature of
    the first evaluation object of each category (72-way). retr_inst_* is harder: f_g
    vs the CLIP features of every evaluation object (N-way over held-out instances).
    """
    out = {
        "n": int(len(e["obj"])),
        "shape_top1": topk(e["shape_logits"], e["cat"], 1),
        "shape_top5": topk(e["shape_logits"], e["cat"], 5),
        "color_top1": topk(e["color_logits"], e["color"], 1),
    }
    first = {}
    for o in eval_objs:
        first.setdefault(int(b.obj_cat[o]), int(o))
    cats = np.array(sorted(first))
    cand = b.clip_video[[first[c] for c in cats]]
    logits = scale * e["f_g"] @ cand.T
    out["retr_cat_top1"] = float((cats[logits.argmax(1)] == e["cat"]).mean())
    top5 = cats[np.argsort(-logits, 1)[:, :5]]
    out["retr_cat_top5"] = float((top5 == e["cat"][:, None]).any(axis=1).mean())
    inst_logits = e["f_g"] @ b.clip_video[eval_objs].T
    pos = np.array([np.flatnonzero(eval_objs == o)[0] for o in e["obj"]])
    out["retr_inst_top1"] = topk(inst_logits, pos, 1)
    out["retr_inst_top5"] = topk(inst_logits, pos, 5)
    out["chance_shape_top1"] = 1.0 / len(cats)
    out["chance_inst_top1"] = 1.0 / len(eval_objs)
    return out


def run_dir(run: str) -> Path:
    p = config.default_runs_root() / run
    p.mkdir(parents=True, exist_ok=True)
    return p


def common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--run", required=True, help="run name; outputs go to <runs-root>/<run>")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--geometry", default="point_cloud_simple")
    ap.add_argument("--split", default="ours", choices=list(config.SPLITS))
    ap.add_argument("--tiny", action="store_true", help="4 categories; overfit / smoke test")
    ap.add_argument("--seed", type=int, default=0)


def setup(args):
    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    cats = [0, 14, 26, 51] if args.tiny else None  # airplane, chair, guitar, ring
    return b, make_splits(b, args.split, categories=cats)


def train(args):
    set_seed(args.seed)
    dev, out_dir = device(), run_dir(args.run)
    b, splits = setup(args)
    (out_dir / "config.json").write_text(json.dumps(vars(args) | {"splits": {k: len(v) for k, v in splits.items()}},
                                                    indent=1, default=str))
    train_ds = EEG3D(b, splits["train"], train=True, seed=args.seed)
    sel_name = "val" if len(splits["val"]) else "test"
    sel_ds = EEG3D(b, splits[sel_name], train=False)
    test_ds = EEG3D(b, splits["test"], train=False)
    bs = min(args.batch_size, len(train_ds))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=2,
                        persistent_workers=True)

    model = build_encoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=(args.epochs + 5) * len(loader),
                                                final_div_factor=10000, pct_start=min(0.3, 2 / args.epochs))
    color_w = torch.tensor(config.COLOR_CLASS_WEIGHTS, device=dev)
    start, best = 0, -1.0
    last = out_dir / "last.pt"
    if last.exists() and not args.restart:
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck.get("epochs", args.epochs) != args.epochs:
            raise SystemExit(f"last.pt was trained with --epochs {ck['epochs']}; resume with the same value "
                             "(the OneCycle schedule depends on it) or pass --restart")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start} (best {sel_name} retr_cat_top1 {best:.4f})")

    log = (out_dir / "metrics.jsonl").open("a")
    for epoch in range(start, args.epochs):
        model.train()
        t0, parts, n = time.time(), {}, 0
        for batch in loader:
            loss, p = loss_fn(model, forward(model, batch, dev), batch, dev, color_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            n += 1
            for k, v in (p | {"loss": loss.item()}).items():
                parts[k] = parts.get(k, 0) + v
        rec = {"epoch": epoch, "sec": round(time.time() - t0, 1), "lr": sched.get_last_lr()[0]}
        rec |= {f"train_{k}": v / n for k, v in parts.items()}
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            scale = model.logit_scale.item()  # upstream multiplies by logit_scale directly (no exp)
            sel = score(embed(model, sel_ds, dev), b, splits[sel_name], scale)
            rec |= {f"{sel_name}_{k}": v for k, v in sel.items()}
            if sel_name != "test":
                rec |= {f"test_{k}": v for k, v in score(embed(model, test_ds, dev), b, splits["test"], scale).items()}
            if sel["retr_cat_top1"] > best:
                best = sel["retr_cat_top1"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": rec}, out_dir / "best.pt")
                rec["saved_best"] = True
            print(f"ep {epoch:3d} loss {rec['train_loss']:.3f} | {sel_name} shape@1 {sel['shape_top1']:.3f} "
                  f"retr@1 {sel['retr_cat_top1']:.3f} inst@1 {sel['retr_inst_top1']:.3f} | {rec['sec']}s")
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "epochs": args.epochs}, last)
    print(f"done. best {sel_name} retr_cat_top1 = {best:.4f}; checkpoints in {out_dir}")


def load_trained(run: str, dev, which="best.pt"):
    model = build_encoder().to(dev)
    ck = torch.load(run_dir(run) / which, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


# ----------------------------------------------------------------------------- diagnostics
def shape_descriptor(xyz: np.ndarray, rng, bins=16, pairs=20000) -> np.ndarray:
    """Rotation-invariant geometry summary: sorted PCA variances + D2 distance histogram."""
    c = xyz - xyz.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1]
    i, j = rng.integers(0, len(xyz), (2, pairs))
    h, _ = np.histogram(np.linalg.norm(xyz[i] - xyz[j], axis=1), bins=bins, range=(0, 2), density=True)
    return np.concatenate([ev / ev.sum(), h / bins])


def within_category_probe(b: Bundle, e_train: dict, e_test: dict, alpha=10.0) -> dict:
    """Does f_g carry instance geometry beyond category?

    Subtract each category's mean from both embeddings and shape descriptors (category means
    estimated on training objects), fit ridge on training residuals, report R^2 on test
    residuals. R^2 near 0 = no within-category shape information is linearly available.
    """
    from numpy.linalg import solve
    rng = np.random.default_rng(0)
    desc = {o: shape_descriptor(b.points[o], rng) for o in np.unique(np.concatenate([e_train["obj"], e_test["obj"]]))}

    def residual(e, mx, my):
        X = np.stack([e["f_g"][i] - mx[e["cat"][i]] for i in range(len(e["obj"]))])
        Y = np.stack([desc[e["obj"][i]] - my[e["cat"][i]] for i in range(len(e["obj"]))])
        return X, Y

    cats = np.unique(e_train["cat"])
    mx = {c: e_train["f_g"][e_train["cat"] == c].mean(0) for c in cats}
    my = {c: np.mean([desc[o] for o in np.unique(e_train["obj"][e_train["cat"] == c])], 0) for c in cats}
    Xtr, Ytr = residual(e_train, mx, my)
    Xte, Yte = residual(e_test, mx, my)
    W = solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    pred = Xte @ W
    r2 = 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    # permutation null: same fit with shuffled training targets
    null = []
    for s in range(20):
        Wp = solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ Ytr[np.random.default_rng(s).permutation(len(Ytr))])
        p = Xte @ Wp
        null.append(1 - ((Yte - p) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())
    return {"probe_r2": float(r2), "probe_null_r2_mean": float(np.mean(null)), "probe_null_r2_max": float(np.max(null))}


def sibling_2afc(b: Bundle, e: dict) -> float:
    """Within-category 2-way: is f_g closer to its own object's CLIP feature than to a sibling's?
    Chance 0.5. Uses every same-category pair present in the split."""
    wins = []
    for i, (o, c) in enumerate(zip(e["obj"], e["cat"])):
        sibs = [s for s in np.unique(e["obj"][e["cat"] == c]) if s != o]
        own = e["f_g"][i] @ b.clip_video[o]
        wins += [own > e["f_g"][i] @ b.clip_video[s] for s in sibs]
    return float(np.mean(wins)) if wins else float("nan")


def diagnose(args):
    dev = device()
    b, splits = setup(args)
    model = load_trained(args.run, dev, args.ckpt)
    scale = model.logit_scale.item()
    test_avg = EEG3D(b, splits["test"], train=False)
    test_trial = EEG3D(b, splits["test"], train=False, average=False)
    train_avg = EEG3D(b, splits["train"], train=False)
    res = {}

    def add(name, ds, **kw):
        res[name] = score(embed(model, ds, dev, **kw), b, splits["test"], scale)
        print(f"{name:28s} shape@1 {res[name]['shape_top1']:.3f}  shape@5 {res[name]['shape_top5']:.3f}  "
              f"retr@1 {res[name]['retr_cat_top1']:.3f}  inst@1 {res[name]['retr_inst_top1']:.3f}")

    print(f"chance: shape@1 {1/72:.3f}  shape@5 {5/72:.3f}  inst@1 {1/len(splits['test']):.3f}")
    add("test_repeats_averaged", test_avg)
    add("test_single_trial", test_trial)
    add("dynamic_only (static=0)", test_avg, stat_scale=0.0)
    add("static_only (dynamic=0)", test_avg, dyn_scale=0.0)
    add("no_eeg (both=0)", test_avg, dyn_scale=0.0, stat_scale=0.0)

    # shuffled EEG: pair each object's label with another object's EEG
    e = embed(model, test_avg, dev)
    perm = np.random.default_rng(0).permutation(len(e["obj"]))
    shuf = {k: (v[perm] if k in ("f_g", "f_a", "shape_logits", "color_logits") else v) for k, v in e.items()}
    res["shuffled_eeg"] = score(shuf, b, splits["test"], scale)
    print(f"{'shuffled_eeg':28s} shape@1 {res['shuffled_eeg']['shape_top1']:.3f}  "
          f"retr@1 {res['shuffled_eeg']['retr_cat_top1']:.3f}")

    # noise robustness on the averaged test EEG
    for k in (0.25, 0.5, 1.0, 2.0):
        ds = EEG3D(b, splits["test"], train=False)
        ds.aug, ds.rng = False, np.random.default_rng(1)
        orig = ds._stream

        def noisy(arr, key, trial, k=k, orig=orig, ds=ds):
            x = orig(arr, key, trial)
            return x + ds.rng.standard_normal(x.shape).astype(np.float32) * x.std(axis=1, keepdims=True) * k
        ds._stream = noisy
        add(f"noise_{k}xstd", ds)

    res["sibling_2afc_test"] = sibling_2afc(b, e)
    print(f"within-category 2AFC (chance 0.5): {res['sibling_2afc_test']:.3f}")
    res |= within_category_probe(b, embed(model, train_avg, dev), e)
    print(f"within-category shape probe R^2 {res['probe_r2']:.3f} "
          f"(null mean {res['probe_null_r2_mean']:.3f}, max {res['probe_null_r2_max']:.3f})")
    path = run_dir(args.run) / "diagnostics.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")


def export(args):
    dev = device()
    b, splits = setup(args)
    model = load_trained(args.run, dev, args.ckpt)
    out = {}
    for name, objs in splits.items():
        if len(objs) == 0:
            continue
        for mode, avg in (("avg", True), ("trial", False)):
            e = embed(model, EEG3D(b, objs, train=False, average=avg), dev)
            out |= {f"{name}_{mode}_{k}": v for k, v in e.items()}
    path = run_dir(args.run) / "embeddings.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}: {sorted(out)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); common_args(t)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--eval-every", type=int, default=1)
    t.add_argument("--restart", action="store_true", help="ignore last.pt")
    for name in ("diagnose", "embed"):
        p = sub.add_parser(name); common_args(p)
        p.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args(argv)
    {"train": train, "diagnose": diagnose, "embed": export}[args.cmd](args)


if __name__ == "__main__":
    main()
