"""Neuro-3D's generation metrics, computed on our outputs and on no-information floors.

The paper scores a generated cloud with a point-cloud category classifier: N-way top-K
accuracy (2-way top-1, 10-way top-3), averaged over 5 samples and as the best of 5.
Their classifier is a PointNet++ trained on Objaverse; ours is a PointNet trained on the 504
training shapes of this dataset (7 per category), so absolute numbers are not identical to
theirs. What is comparable is every row scored by the SAME classifier, including floors.

N-way top-K is computed exactly, not by sampling distractors: if r of the 71 other
categories outscore the true one, the chance that fewer than K of N-1 random distractors
are among those r is a hypergeometric CDF. 2-way top-1 reduces to 1 - r/71.

    python -m n3d.parity train-cls --run cls_shapes
    python -m n3d.parity score --cls-run cls_shapes --runs pn_film pn_category pn_oracle_clip
    python -m n3d.parity score --cls-run cls_shapes --runs pn_film --file pred_test_k5_s100_zeros.npz

`score` always adds reference rows that need no generator:
  ground_truth        the real test shapes (the classifier's ceiling)
  category_medoid     the most typical training shape of the TRUE category
  global_medoid       one fixed training shape for every input   (floor)
  random_train_shape  a random training shape per input           (floor)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .ckpt import atomic_save
from .data import load_bundle, make_splits
from .encoder import run_dir


class PointNetCls(nn.Module):
    def __init__(self, n_classes: int = config.N_CATEGORIES, width: int = 256):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                   nn.Linear(128, width), nn.ReLU(), nn.Linear(width, 2 * width))
        self.head = nn.Sequential(nn.Linear(4 * width, width), nn.ReLU(), nn.Dropout(0.4), nn.Linear(width, n_classes))

    def forward(self, x):  # (B, N, 3)
        h = self.point(x)
        return self.head(torch.cat([h.amax(1), h.mean(1)], 1))


def augment(x: torch.Tensor, n: int) -> torch.Tensor:
    """Random subsample, anisotropic scale, small rotation, jitter. Jitter up to 0.05 so the
    classifier tolerates the fuzz of generated clouds."""
    B = x.shape[0]
    idx = torch.rand(B, x.shape[1], device=x.device).argsort(1)[:, :n]
    x = torch.gather(x, 1, idx[..., None].expand(-1, -1, 3))
    x = x * (1 + 0.15 * (torch.rand(B, 1, 3, device=x.device) * 2 - 1))
    a = (torch.rand(B, device=x.device) * 2 - 1) * 0.26  # +-15 degrees about the up axis
    c, s, o, z = a.cos(), a.sin(), torch.ones_like(a), torch.zeros_like(a)
    R = torch.stack([c, z, s, z, o, z, -s, z, c], 1).view(B, 3, 3)
    x = x @ R
    return x + torch.rand(B, 1, 1, device=x.device) * 0.05 * torch.randn_like(x)


@torch.no_grad()
def logits_of(model, clouds: np.ndarray, dev, n: int, seed: int = 0, bs: int = 64) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, len(clouds), bs):
        x = clouds[i:i + bs]
        x = x[:, rng.choice(x.shape[1], min(n, x.shape[1]), replace=False)]
        out.append(model(torch.from_numpy(x).float().to(dev)).cpu().numpy())
    return np.concatenate(out)


def nway(logits: np.ndarray, y: np.ndarray, n: int, k: int) -> np.ndarray:
    """Exact probability that the true class is in the top-k among itself and n-1 random distractors."""
    from scipy.stats import hypergeom
    C = logits.shape[1]
    r = (logits > logits[np.arange(len(y)), y][:, None]).sum(1)
    return hypergeom.cdf(k - 1, C - 1, r, n - 1)


def summarize(logits: np.ndarray, y: np.ndarray) -> dict:
    """logits (N, K_samples, C). Mean over samples, and best sample per object (selected with
    the ground truth, as in the paper's best-of-5; not achievable by a real decoder)."""
    N, K, C = logits.shape
    yy = np.repeat(y, K)
    flat = logits.reshape(N * K, C)
    out = {}
    for name, (n, k) in {"2way_top1": (2, 1), "10way_top3": (10, 3)}.items():
        p = nway(flat, yy, n, k).reshape(N, K)
        out[name + "_avg"] = float(p.mean())
        out[name + "_best"] = float(p.max(1).mean())
    hit = (flat.argmax(1) == yy).reshape(N, K)
    out["72way_top1_avg"] = float(hit.mean())
    out["72way_top1_best"] = float(hit.any(1).mean())
    return out


def cmd_train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    b = load_bundle(args.data_root, ("sub01",), args.geometry)
    s = make_splits(b, "ours")
    X = torch.from_numpy(b.points[s["train"]]).float().to(dev)
    Y = torch.from_numpy(b.obj_cat[s["train"]]).to(dev)
    model = PointNetCls().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    for step in range(1, args.steps + 1):
        model.train()
        i = torch.randint(0, len(X), (args.batch_size,), device=dev)
        loss = F.cross_entropy(model(augment(X[i], args.points)), Y[i], label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            acc = {k: float((logits_of(model, b.points[s[k]], dev, args.points).argmax(1) == b.obj_cat[s[k]]).mean())
                   for k in ("train", "val", "test")}
            print(f"step {step:5d} loss {loss.item():.3f} | real-shape top1: train {acc['train']:.3f} val {acc['val']:.3f} test {acc['test']:.3f}")
    out = run_dir(args.run)
    atomic_save({"model": model.state_dict(), "points": args.points, "geometry": args.geometry, "real_shape_acc": acc}, out / "cls.pt")
    (out / "cls.json").write_text(json.dumps({"real_shape_top1": acc, "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    print(f"wrote {out / 'cls.pt'}")


def cmd_score(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(run_dir(args.cls_run) / "cls.pt", map_location=dev, weights_only=False)
    model = PointNetCls().to(dev)
    model.load_state_dict(ck["model"])
    n = ck["points"]
    b = load_bundle(args.data_root, ("sub01",), ck["geometry"])
    s = make_splits(b, "ours")
    rows, ref_objs = {}, None
    for r in args.runs:
        z = np.load(run_dir(r) / args.file)
        pred = z["pred"] if z["pred"].ndim == 4 else z["pred"][:, None]
        objs = z["obj"]
        ref_objs = objs if ref_objs is None else ref_objs
        N, K = pred.shape[:2]
        lg = logits_of(model, pred.reshape(N * K, -1, 3), dev, n).reshape(N, K, -1)
        rows[f"{r}:{args.file}"] = summarize(lg, b.obj_cat[objs]) | {"samples": int(K)}
    objs = ref_objs if ref_objs is not None else s["test"]
    y = b.obj_cat[objs]
    from .baselines import global_medoid, medoids
    med = medoids(b, s["train"])
    gmed = global_medoid(b, s["train"], np.array(sorted(med.values())))
    rng = np.random.default_rng(0)
    refs = {"ground_truth": b.points[objs][:, None],
            "category_medoid (true category)": np.stack([b.points[med[int(c)]] for c in y])[:, None],
            "global_medoid (floor)": np.repeat(b.points[gmed][None, None], len(objs), 0),
            "random_train_shape (floor)": b.points[rng.choice(s["train"], (len(objs), 5))]}
    for name, clouds in refs.items():
        N, K = clouds.shape[:2]
        lg = logits_of(model, clouds.reshape(N * K, -1, 3), dev, n).reshape(N, K, -1)
        rows[name] = summarize(lg, y) | {"samples": int(K)}
    rows["chance"] = {"2way_top1_avg": 0.5, "10way_top3_avg": 0.3, "72way_top1_avg": 1 / 72}
    rows["Neuro-3D paper (their classifier, sub01-05)"] = {"2way_top1_avg": 0.5581, "2way_top1_best": 0.7208,
                                                          "10way_top3_avg": 0.3589, "10way_top3_best": 0.5764}
    out = run_dir(args.cls_run) / f"parity_{args.out_tag}.json"
    out.write_text(json.dumps({"classifier_real_shape_top1": ck.get("real_shape_acc"), "rows": rows}, indent=1))
    cols = ["2way_top1_avg", "2way_top1_best", "10way_top3_avg", "10way_top3_best", "72way_top1_avg"]
    print(f"{'':52s}" + " ".join(f"{c:>16s}" for c in cols))
    for name, m in rows.items():
        print(f"{name[:52]:52s}" + " ".join(f"{m[c]:16.3f}" if c in m else f"{'':>16s}" for c in cols))
    print(f"classifier top1 on real shapes: {ck.get('real_shape_acc')}\nwrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-cls")
    t.add_argument("--run", required=True)
    t.add_argument("--data-root", type=Path, default=None)
    t.add_argument("--geometry", default="point_cloud_simple")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--points", type=int, default=2048)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--cls-run", required=True)
    sc.add_argument("--runs", nargs="*", default=[])
    sc.add_argument("--file", default="pred_test_k5_s100.npz")
    sc.add_argument("--data-root", type=Path, default=None)
    sc.add_argument("--out-tag", default="test")
    args = ap.parse_args(argv)
    {"train-cls": cmd_train, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
