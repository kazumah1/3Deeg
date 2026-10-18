"""Write a fake packed dataset with the real shapes, for CPU smoke tests only.

    python -m n3d.synthetic /tmp/n3d_fake
EEG carries a weak category signal so training curves move. Not real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import config


def make(root: Path, participants=("sub01",), seed=0):
    rng = np.random.default_rng(seed)
    packed = root / "packed"
    packed.mkdir(parents=True, exist_ok=True)
    cats = [f"{i + 1:02d}_cat{i:02d}" for i in range(config.N_CATEGORIES)]
    objects = [f"{c}_{v:02d}" for c in cats for v in range(config.N_VARIANTS)]
    (packed / "objects.json").write_text(json.dumps({"objects": objects, "categories": cats}))
    geo = np.zeros((len(objects), config.N_POINTS, 6), np.float32)
    for i in range(len(objects)):
        c = i // config.N_VARIANTS
        u = rng.standard_normal((config.N_POINTS, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        geo[i, :, :3] = u * (1 + 0.5 * np.array([np.sin(c), np.cos(c), (c % 5) / 5]) + 0.05 * rng.standard_normal(3))
        geo[i, :, 3:] = rng.random(3)
    for name in ("point_cloud", "point_cloud_simple"):
        np.save(packed / f"geometry_{name}.npy", geo)
    clip = rng.standard_normal((config.N_CATEGORIES, 1024)).repeat(config.N_VARIANTS, 0) * 0.05
    np.save(packed / "clip_video.npy", (clip + 0.01 * rng.standard_normal(clip.shape)).astype(np.float32))
    np.save(packed / "color_label.npy", rng.integers(0, 6, len(objects)))
    pattern_d = rng.standard_normal((config.N_CATEGORIES, 64, 1)) * 0.5
    for sub in participants:
        labels = []
        for o in objects:
            labels += [o] * (4 if o.endswith(("08", "09")) else 2)
        labels = np.array(labels)
        rng.shuffle(labels)
        cat = np.array([objects.index(l) // config.N_VARIANTS for l in labels])
        dyn = rng.standard_normal((len(labels), 64, config.DYN_LEN)).astype(np.float32) + pattern_d[cat]
        stat = rng.standard_normal((len(labels), 64, config.STAT_LEN)).astype(np.float32) + pattern_d[cat]
        np.savez(packed / f"{sub}_eeg.npz", dyn=dyn, stat=stat, label=labels)
    return root


if __name__ == "__main__":
    print(make(Path(sys.argv[1])))
