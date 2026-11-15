"""Frozen geometry-embedding extraction and cache validation."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def save_embedding_cache(path: Path, labels: np.ndarray, fg: np.ndarray) -> None:
    labels, fg = np.asarray(labels).astype(str), np.asarray(fg, dtype=np.float32)
    if fg.ndim != 2 or fg.shape[1] != 1024 or len(labels) != len(fg):
        raise ValueError("Expected one 1024-D f_g row per object")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels, fg=fg)


def load_embedding_cache(path: Path, labels: list[str] | tuple[str, ...]) -> np.ndarray:
    """Load object-level, repeat-averaged frozen f_g values in requested order."""
    data = np.load(path, allow_pickle=False)
    by_label = {str(label): feature for label, feature in zip(data["labels"].astype(str), data["fg"])}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(f"Frozen embedding cache lacks {len(missing)} requested objects, e.g. {missing[:3]}")
    values = np.stack([by_label[label] for label in labels]).astype(np.float32)
    if values.shape[1] != 1024:
        raise ValueError(f"Expected f_g dimension 1024, got {values.shape[1]}")
    return values
