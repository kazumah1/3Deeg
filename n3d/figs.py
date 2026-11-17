"""Static figures for the poster (matplotlib PNG, no plotly).

gallery: rows are test objects picked at fixed quantiles of the first model's per-object
Chamfer (not cherry-picked: 10th..90th percentile), columns are ground truth, each model's
first sample, and the true category's medoid training shape (the "knows only the category"
reference). Each panel is titled with its Chamfer to ground truth.

    python -m n3d.figs gallery --runs bench_clip_pn_concat_s0 bench_clip_pv_concat_s0 \
        --labels "PointNet 1.4M" "PVCNN 29M (Neuro-3D)" --out figs/gallery_clip.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .baselines import medoids
from .data import load_bundle, make_splits
from .encoder import run_dir
from .metrics import chamfer_np

CD_KEY = "cd"


def _cd(a, b) -> float:
    r = chamfer_np(a, b)
    return r[CD_KEY]


def _panel(ax, pts, title, n=2048, seed=0):
    p = pts[np.random.default_rng(seed).choice(len(pts), min(n, len(pts)), replace=False)]
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=0.6, c=p[:, 1], cmap="viridis", linewidths=0)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def gallery(runs, labels, out: Path, file="pred_test_k5_s100.npz", quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = load_bundle(None, ("sub01",))
    med = medoids(b, make_splits(b, "ours")["train"])
    zs = [np.load(run_dir(r) / file) for r in runs]
    objs, gt = zs[0]["obj"], zs[0]["gt"]
    preds = [z["pred"][:, 0] for z in zs]
    for z in zs[1:]:
        assert np.array_equal(z["obj"], objs), "runs must share the test object order"
    cd0 = np.array([_cd(preds[0][i], gt[i]) for i in range(len(objs))])
    order = np.argsort(cd0)
    rows = [int(order[int(q * (len(order) - 1))]) for q in quantiles]

    cols = ["ground truth", *labels, "category medoid"]
    fig = plt.figure(figsize=(2.1 * len(cols), 2.1 * len(rows)))
    for r, i in enumerate(rows):
        cat = int(b.obj_cat[objs[i]])
        m = b.points[med[cat]]
        clouds = [gt[i], *[p[i] for p in preds], m]
        for c, (pts, name) in enumerate(zip(clouds, cols)):
            ax = fig.add_subplot(len(rows), len(cols), r * len(cols) + c + 1, projection="3d")
            t = name if r == 0 else ""
            if c == 0:
                t = (t + "\n" if t else "") + b.categories[cat]
            else:
                t = (t + "\n" if t else "") + f"CD {_cd(pts, gt[i]):.3f}"
            _panel(ax, pts, t)
    fig.suptitle(f"CLIP-conditioned samples, test objects at Chamfer quantiles {quantiles} of {labels[0]}", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gallery")
    g.add_argument("--runs", nargs="+", required=True)
    g.add_argument("--labels", nargs="+", required=True)
    g.add_argument("--file", default="pred_test_k5_s100.npz")
    g.add_argument("--out", type=Path, default=Path("figs/gallery.png"))
    a = ap.parse_args(argv)
    if a.cmd == "gallery":
        assert len(a.runs) == len(a.labels)
        gallery(a.runs, a.labels, a.out, a.file)


if __name__ == "__main__":
    main()
