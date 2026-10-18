"""Arrays, splits and torch datasets.

Object ordering follows upstream: the 720 sorted labels ('01_airplane_00', ...),
so category index = position in sorted category order and matches upstream
class labels. Splits are by object variant, so no object appears on both sides.

Faithful to upstream EEGdataset.AllDataFeatureTwoEEG:
  * point clouds centred and scaled to unit max radius (pc_norm)
  * training augmentation: with p=0.25 a stream is replaced by the mean of that
    object's repeats; with p=0.6 Gaussian noise of 0.2 x per-channel std is added
  * evaluation averages an object's repeats into one sample (test_mean=True)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


def pc_norm(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    return xyz / np.sqrt((xyz ** 2).sum(axis=1)).max()


@dataclass
class Bundle:
    """Everything loaded once and shared by datasets."""
    objects: list[str]          # 720 names
    categories: list[str]       # 72 names
    obj_cat: np.ndarray         # (720,) category index
    obj_variant: np.ndarray     # (720,) '00'..'09'
    points: np.ndarray          # (720, 8192, 3) normalised XYZ
    clip_video: np.ndarray      # (720, 1024)
    color: np.ndarray           # (720,)
    dyn: np.ndarray             # (T, 64, 600)
    stat: np.ndarray            # (T, 64, 250)
    trial_obj: np.ndarray       # (T,) object index
    trial_sub: np.ndarray       # (T,) participant index

    def objects_in(self, variants: list[str], categories: list[int] | None = None) -> np.ndarray:
        keep = np.isin(self.obj_variant, variants)
        if categories is not None:
            keep &= np.isin(self.obj_cat, categories)
        return np.flatnonzero(keep)


def load_bundle(data_root: Path | None = None, participants=("sub01",), geometry="point_cloud_simple") -> Bundle:
    packed = Path(data_root or config.default_data_root()) / "packed"
    meta = json.loads((packed / "objects.json").read_text())
    objects, categories = meta["objects"], meta["categories"]
    index = {n: i for i, n in enumerate(objects)}
    obj_cat = np.array([categories.index(n.rsplit("_", 1)[0]) for n in objects])
    obj_variant = np.array([n.rsplit("_", 1)[1] for n in objects])

    geo = np.load(packed / f"geometry_{geometry}.npy", mmap_mode="r")
    points = np.stack([pc_norm(np.asarray(g[:, :3], np.float64)) for g in geo]).astype(np.float32)

    dyn, stat, tobj, tsub = [], [], [], []
    for si, sub in enumerate(participants):
        z = np.load(packed / f"{sub}_eeg.npz")
        dyn.append(z["dyn"])
        stat.append(z["stat"])
        tobj.append(np.array([index[str(x)] for x in z["label"]]))
        tsub.append(np.full(len(z["label"]), si))
    return Bundle(objects, categories, obj_cat, obj_variant, points,
                  np.load(packed / "clip_video.npy").astype(np.float32),
                  np.load(packed / "color_label.npy").astype(np.int64),
                  np.concatenate(dyn), np.concatenate(stat), np.concatenate(tobj), np.concatenate(tsub))


class EEG3D(Dataset):
    """One sample per trial (train=True) or per (participant, object) with repeats averaged.

    Parameters
    ----------
    objects : object indices allowed in this split
    train   : apply upstream augmentation and return single trials
    average : for evaluation, average all repeats of an object (upstream default)
    """

    def __init__(self, b: Bundle, objects: np.ndarray, train: bool, average: bool = True,
                 aug: bool | None = None, seed: int = 0):
        self.b, self.train = b, train
        self.aug = train if aug is None else aug
        self.rng = np.random.default_rng(seed)
        allowed = np.isin(b.trial_obj, objects)
        self.groups: dict[tuple[int, int], np.ndarray] = {}
        for t in np.flatnonzero(allowed):
            self.groups.setdefault((int(b.trial_sub[t]), int(b.trial_obj[t])), []).append(t)
        self.groups = {k: np.array(v) for k, v in sorted(self.groups.items())}
        if train or not average:
            self.items = [(k, t) for k, ts in self.groups.items() for t in ts]
        else:
            self.items = [(k, None) for k in self.groups]

    def __len__(self):
        return len(self.items)

    def _stream(self, arr: np.ndarray, key, trial) -> np.ndarray:
        reps = self.groups[key]
        if trial is None:
            x = arr[reps].mean(axis=0)
        elif self.aug and self.rng.random() > 0.75:
            x = arr[reps].mean(axis=0)
        else:
            x = arr[trial]
        x = np.array(x, np.float32, copy=True)
        if self.aug and self.rng.random() > 0.4:
            x += self.rng.standard_normal(x.shape).astype(np.float32) * x.std(axis=1, keepdims=True) * 0.2
        return x

    def __getitem__(self, i):
        (sub, obj), trial = self.items[i]
        b = self.b
        return {
            "dyn": torch.from_numpy(self._stream(b.dyn, (sub, obj), trial)),
            "stat": torch.from_numpy(self._stream(b.stat, (sub, obj), trial)),
            "cat": int(b.obj_cat[obj]),
            "color": int(b.color[obj]),
            "obj": obj,
            "sub": sub,
            "points": torch.from_numpy(b.points[obj]),
            "clip": torch.from_numpy(b.clip_video[obj]),
        }


def make_splits(b: Bundle, split: str = "ours", categories: list[int] | None = None,
                max_objects_per_split: int | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Object indices per split. `categories` / `max_objects_per_split` make tiny debug subsets."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, variants in config.SPLITS[split].items():
        objs = b.objects_in(variants, categories)
        if max_objects_per_split is not None and len(objs) > max_objects_per_split:
            objs = np.sort(rng.choice(objs, max_objects_per_split, replace=False))
        out[name] = objs
    return out
