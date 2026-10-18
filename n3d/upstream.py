"""Access to the upstream Neuro-3D code at a pinned commit.

The upstream repo has no licence file, so we do not copy it into this repo. It
is cloned at runtime and its model classes are imported unchanged. Only the
encoder (VideoImageEEGClassifyColor3) and PVCNN2 are used; data loading,
training loops and diffusion scheduling are reimplemented in this package so we
control splits, checkpointing and evaluation.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config


def upstream_dir() -> Path:
    base = Path(os.environ.get("N3D_UPSTREAM", Path.home() / ".cache" / "n3d"))
    return base / f"neuro-3D-{config.UPSTREAM_COMMIT[:10]}"


def ensure_upstream() -> Path:
    path = upstream_dir()
    if not (path / "eeg_data_process" / "extract_eeg_feature.py").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", config.UPSTREAM_URL, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", config.UPSTREAM_COMMIT], check=True)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def encoder_class():
    """The upstream dynamic-static fusion encoder."""
    ensure_upstream()
    from eeg_data_process.extract_eeg_feature import VideoImageEEGClassifyColor3
    return VideoImageEEGClassifyColor3


def build_encoder():
    cls = encoder_class()
    return cls(num_channels=config.EEG_CHANNELS, sequence_length=config.DYN_LEN,
               sequence_length2=config.STAT_LEN, num_latents=1024, cls_num=config.N_CATEGORIES)


def pvcnn2_class():
    """PVCNN2 from upstream. Importing it JIT-compiles CUDA ops (needs nvcc; ~3-8 min once).

    The build is cached in TORCH_EXTENSIONS_DIR; on Colab point it at Drive so a
    new runtime with the same torch/CUDA reuses it.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PVCNN needs a CUDA GPU; use --denoiser pointnet on CPU")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    ensure_upstream()
    from PointGeneration.pvcnn.pvcnn import PVCNN2
    return PVCNN2
