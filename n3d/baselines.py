"""Reference Chamfer scores that any EEG-to-3D model must be read against. CPU only.

Geometry-only (no EEG):
  random_other_category   CD to a random object of another category      (chance ceiling)
  same_category_mean      mean CD to the category's training objects     (what category alone buys)
  category_medoid         CD to the category's medoid training object    (best fixed template per category)
  nearest_train_same_cat  CD to the closest training object of the category (oracle retrieval)
With an encoder run (uses its embeddings.npz from `n3d.encoder embed`):
  classify_then_template  predict category from EEG, output that category's medoid
  retrieve_then_copy      nearest training object by f_g . CLIP(video), copied as the output
A generative model that does not beat classify_then_template is not using shape
information beyond the category it implicitly predicts.

    python -m n3d.baselines --run baselines_sub01
    python -m n3d.baselines --run baselines_sub01 --encoder-run enc_sub01
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import config
from .data import load_bundle, make_splits
from .encoder import run_dir
from .metrics import chamfer_np


def medoids(b, train_objs) -> dict[int, int]:
    out = {}
    for c in np.unique(b.obj_cat[train_objs]):
        objs = train_objs[b.obj_cat[train_objs] == c]
        D = np.zeros((len(objs), len(objs)))
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                D[i, j] = D[j, i] = chamfer_np(b.points[objs[i]], b.points[objs[j]])["cd"]
        out[int(c)] = int(objs[D.sum(1).argmin()])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--geometry", default="point_cloud_simple")
    ap.add_argument("--split", default="ours", choices=list(config.SPLITS))
    ap.add_argument("--encoder-run", default=None)
    args = ap.parse_args(argv)

    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    s = make_splits(b, args.split)
    train, test = s["train"], s["test"]
    rng = np.random.default_rng(0)
    med = medoids(b, train)
    rows = []
    for o in test:
        c = int(b.obj_cat[o])
        others = train[b.obj_cat[train] != c]
        same = train[b.obj_cat[train] == c]
        same_cd = [chamfer_np(b.points[o], b.points[t])["cd"] for t in same]
        rows.append({
            "obj": int(o), "name": b.objects[o], "category": b.categories[c],
            "random_other_category": float(np.mean([chamfer_np(b.points[o], b.points[t])["cd"]
                                                    for t in rng.choice(others, 3, replace=False)])),
            "same_category_mean": float(np.mean(same_cd)),
            "category_medoid": chamfer_np(b.points[o], b.points[med[c]])["cd"],
            "nearest_train_same_cat": float(np.min(same_cd)),
        })

    if args.encoder_run:
        e = np.load(run_dir(args.encoder_run) / "embeddings.npz")
        pred_cat = {int(o): int(l.argmax()) for o, l in zip(e["test_avg_obj"], e["test_avg_shape_logits"])}
        train_feats = b.clip_video[train]
        f_g = {int(o): f for o, f in zip(e["test_avg_obj"], e["test_avg_f_g"])}
        for r in rows:
            r["pred_category"] = b.categories[pred_cat[r["obj"]]]
            r["classify_then_template"] = chamfer_np(b.points[r["obj"]], b.points[med[pred_cat[r["obj"]]]])["cd"]
            nn = int(train[np.argmax(train_feats @ f_g[r["obj"]])])
            r["retrieve_then_copy"] = chamfer_np(b.points[r["obj"]], b.points[nn])["cd"]

    keys = [k for k in rows[0] if k not in ("obj", "name", "category", "pred_category")]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows]))}
               for k in keys}
    summary["n_test_objects"] = len(rows)
    summary["metric"] = "symmetric mean squared Chamfer on pc_norm clouds (n3d.metrics)"
    out = run_dir(args.run)
    with (out / "baselines_per_object.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "baselines_summary.json").write_text(json.dumps(summary, indent=1))
    for k in keys:
        print(f"{k:26s} mean {summary[k]['mean']:.5f}  median {summary[k]['median']:.5f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
