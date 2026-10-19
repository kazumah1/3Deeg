"""Crash-safe checkpointing for Colab + Google Drive.

Three layers, so a runtime that dies at any moment loses little:
  last.pt        rolling resume point, rewritten often. Written to a temp file and renamed
                 into place, so a kill mid-write never leaves a half-written last.pt.
  last_prev.pt   the previous last.pt, kept in case the newest one never finished
                 uploading to Drive before the runtime died.
  snapshots/     frozen full checkpoints (model + optimizer + scheduler) at fixed
                 intervals; the newest `keep` are kept. Resume falls back to them.
Logs are appended with open/close per line: Drive only uploads a file once it is closed,
so a log held open for hours is not backed up until the run ends.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def atomic_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_resume(obj, run_dir: Path) -> None:
    last, prev = run_dir / "last.pt", run_dir / "last_prev.pt"
    tmp = run_dir / "last.pt.tmp"
    torch.save(obj, tmp)
    if last.exists():
        os.replace(last, prev)
    os.replace(tmp, last)


def save_snapshot(obj, run_dir: Path, name: str, keep: int) -> Path:
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    path = snap_dir / name
    atomic_save(obj, path)
    if keep > 0:
        snaps = sorted(snap_dir.glob("*.pt"))
        for old in snaps[:-keep]:
            old.unlink(missing_ok=True)
    return path


def load_resume(run_dir: Path, map_location):
    """Newest loadable full checkpoint: last.pt, then last_prev.pt, then snapshots newest first."""
    candidates = [run_dir / "last.pt", run_dir / "last_prev.pt"]
    candidates += sorted((run_dir / "snapshots").glob("*.pt"), reverse=True) if (run_dir / "snapshots").exists() else []
    for path in candidates:
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e:  # truncated upload, partial write from an older version
            print(f"skipping unreadable checkpoint {path.name}: {type(e).__name__}: {e}")
            continue
        if path.name != "last.pt":
            print(f"resuming from fallback checkpoint {path.relative_to(run_dir)}")
        return ck
    return None


def log_line(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
