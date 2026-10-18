"""Shared constants. Everything that changes a result is pinned here."""
from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "guozq21/neuro-3D"
HF_REVISION = "31ebd72706a6b6b40bfe77022f9ca48a7a12dfdd"  # same pin as scripts/setup_neuro3d.py

UPSTREAM_URL = "https://github.com/gzq17/neuro-3D"
UPSTREAM_COMMIT = "277b9cab952dacfaa70e069e883d0b4d07a38180"  # main on 2025-04-03

N_CATEGORIES = 72
N_VARIANTS = 10
N_POINTS = 8192
EEG_CHANNELS = 64
DYN_LEN = 600   # process_data_6s_100Hz.npy, rotating video
STAT_LEN = 250  # process_data_1s_250Hz.npy, static image

# Upstream holds out variants 08 and 09 for testing and selects checkpoints on that
# same test set. We carve a validation split from training so model selection never
# sees test objects. "upstream" reproduces the original split exactly.
SPLITS = {
    "ours": {"train": [f"{i:02d}" for i in range(0, 7)], "val": ["07"], "test": ["08", "09"]},
    "upstream": {"train": [f"{i:02d}" for i in range(0, 8)], "val": [], "test": ["08", "09"]},
}

# Upstream color classifier class weights (classification/retri_shape_color.py).
COLOR_CLASS_WEIGHTS = [1.0, 1.0, 0.1, 0.1, 1.0, 1.0]


def default_data_root() -> Path:
    """Drive in Colab, ./data/neuro-3d elsewhere. Override with N3D_DATA."""
    if os.environ.get("N3D_DATA"):
        return Path(os.environ["N3D_DATA"])
    drive = Path("/content/drive/MyDrive/3deeg/data/neuro-3d")
    if drive.parent.parent.exists():
        return drive
    return Path(__file__).resolve().parents[1] / "data" / "neuro-3d"


def default_runs_root() -> Path:
    """Checkpoints and results. Override with N3D_RUNS."""
    if os.environ.get("N3D_RUNS"):
        return Path(os.environ["N3D_RUNS"])
    drive = Path("/content/drive/MyDrive/3deeg/runs")
    if drive.parent.parent.exists():
        return drive
    return Path(__file__).resolve().parents[1] / "runs"
