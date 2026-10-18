"""Download the Neuro-3D files this project needs and pack them for fast loading.

Runs in Colab (it needs huggingface.co). Archives are kept on Drive so they are
downloaded once; extraction goes to fast local disk; the packed arrays used by
every experiment are written back to Drive.

    python -m n3d.download --participants sub01
    python -m n3d.download --participants sub01 sub02 sub03 sub04 sub05

Packed layout under <data-root>/packed/:
    objects.json                 720 object names in canonical order + category names
    geometry_<source>.npy        (720, 8192, 6) float32, raw XYZ+RGB as distributed
    clip_video.npy               (720, 1024) float32, CLIP video features (training target only)
    color_label.npy              (720,) int64, dominant-colour class (0-5)
    <sub>_eeg.npz                dyn (T,64,600), stat (T,64,250) float32, label (T,) str
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from . import config

GEOMETRY_ARCHIVES = {"point_cloud": "point_cloud.zip", "point_cloud_simple": "point_cloud_simple.zip"}


def _hub():
    from huggingface_hub import HfApi, hf_hub_download
    return HfApi(), hf_hub_download


def list_remote(revision: str = config.HF_REVISION) -> list[str]:
    api, _ = _hub()
    return sorted(api.list_repo_files(config.HF_REPO, repo_type="dataset", revision=revision))


def fetch(name: str, archives: Path, revision: str = config.HF_REVISION) -> Path:
    """Download one repo file into <archives>/<name> (skipped if already present)."""
    _, hf_hub_download = _hub()
    target = archives / name
    if target.exists() and target.stat().st_size > 0:
        print(f"have      {name} ({target.stat().st_size / 1e6:.0f} MB)")
        return target
    print(f"download  {name}")
    path = hf_hub_download(config.HF_REPO, name, repo_type="dataset", revision=revision,
                           local_dir=str(archives))
    return Path(path)


def _find(files: list[str], needle: str) -> str:
    hits = [f for f in files if needle in f.lower()]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one repo file matching {needle!r}, found {hits}. "
                                f"Root files: {[f for f in files if '/' not in f]}")
    return hits[0]


def extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            p = Path(member.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"unsafe member {member.filename}")
        zf.extractall(dest)


def canonical_objects(label_file: Path) -> list[str]:
    """Sorted unique trial labels, e.g. '01_airplane_00'. Matches upstream ordering."""
    labels = np.load(label_file)
    names = sorted({str(x) for x in labels})
    if len(names) != config.N_CATEGORIES * config.N_VARIANTS:
        raise ValueError(f"expected 720 objects, found {len(names)}")
    return names


def pack_geometry(names: list[str], src_dir: Path, out: Path) -> None:
    geo = np.zeros((len(names), config.N_POINTS, 6), np.float32)
    for i, n in enumerate(names):
        arr = np.load(src_dir / f"{n.split('_', 1)[1]}.npy")
        if arr.shape != (config.N_POINTS, 6):
            raise ValueError(f"{n}: unexpected shape {arr.shape}")
        geo[i] = arr
    np.save(out, geo)


def pack_clip(names: list[str], clip_path: Path, out: Path) -> None:
    import torch
    feats = torch.load(clip_path, map_location="cpu", weights_only=False)
    video = np.stack([np.asarray(feats[n.split("_", 1)[1]]["video"], np.float32).reshape(-1) for n in names])
    if video.shape != (len(names), 1024):
        raise ValueError(f"unexpected CLIP video shape {video.shape}")
    np.save(out, video)


def pack_color(names: list[str], xlsx: Path, out: Path) -> None:
    import pandas as pd
    df = pd.read_excel(xlsx)
    table = {str(r["name"]): int(r["label"]) for _, r in df.iterrows()}
    np.save(out, np.array([table[n.split("_", 1)[1]] for n in names], np.int64))


def pack_eeg(sub_dir: Path, out: Path) -> None:
    dyn = np.load(sub_dir / "process_data_6s_100Hz.npy").astype(np.float32)
    stat = np.load(sub_dir / "process_data_1s_250Hz.npy").astype(np.float32)
    label = np.load(sub_dir / "name_label.npy").astype(str)
    if dyn.shape[1:] != (64, 600) or stat.shape[1:] != (64, 250) or not np.isfinite(dyn).all():
        raise ValueError(f"unexpected EEG arrays in {sub_dir}: {dyn.shape} {stat.shape}")
    np.savez(out, dyn=dyn, stat=stat, label=label)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=config.default_data_root())
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--geometry", nargs="+", default=["point_cloud_simple", "point_cloud"],
                    choices=list(GEOMETRY_ARCHIVES))
    ap.add_argument("--scratch", type=Path, default=Path("/content/n3d_extract"),
                    help="fast local disk for extraction (deleted afterwards)")
    ap.add_argument("--list", action="store_true", help="only print the repo file list")
    args = ap.parse_args(argv)

    files = list_remote()
    if args.list:
        print("\n".join(files))
        return
    archives, packed = args.data_root / "archives", args.data_root / "packed"
    archives.mkdir(parents=True, exist_ok=True)
    packed.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch if args.scratch.parent.exists() else args.data_root / "_extract"
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        names = None
        for sub in args.participants:
            out = packed / f"{sub}_eeg.npz"
            zpath = fetch(f"EEGdata/{sub}.zip", archives)
            if not out.exists() or names is None:
                extract(zpath, scratch)
                sub_dir = next(p for p in [scratch / sub, scratch / "EEGdata" / sub] if p.exists())
                if names is None:
                    names = canonical_objects(sub_dir / "name_label.npy")
                if not out.exists():
                    pack_eeg(sub_dir, out)
                    print(f"packed    {out.name}")

        objects_json = packed / "objects.json"
        if objects_json.exists():
            stored = json.loads(objects_json.read_text())["objects"]
            if names is not None and stored != names:
                raise ValueError("object list differs from the packed objects.json")
            names = stored
        else:
            cats = sorted({n.rsplit("_", 1)[0] for n in names})
            objects_json.write_text(json.dumps({"objects": names, "categories": cats}, indent=1))

        for src in args.geometry:
            out = packed / f"geometry_{src}.npy"
            if out.exists():
                continue
            extract(fetch(GEOMETRY_ARCHIVES[src], archives), scratch)
            pack_geometry(names, scratch / src, out)
            print(f"packed    {out.name}")

        if not (packed / "clip_video.npy").exists():
            pack_clip(names, fetch(_find(files, "clip_feature.pth"), archives), packed / "clip_video.npy")
            print("packed    clip_video.npy")
        if not (packed / "color_label.npy").exists():
            pack_color(names, fetch(_find(files, "color_label"), archives), packed / "color_label.npy")
            print("packed    color_label.npy")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"done: {packed}")


if __name__ == "__main__":
    main()
