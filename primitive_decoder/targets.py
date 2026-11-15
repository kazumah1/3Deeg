"""Loading and deterministic preparation of K=3 cuboid targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .geometry import K, tokens_from_components


@dataclass(frozen=True)
class CuboidTargets:
    labels: np.ndarray
    tokens: np.ndarray  # (objects, 3, 12)

    def by_label(self) -> dict[str, np.ndarray]:
        return {str(label): self.tokens[index] for index, label in enumerate(self.labels)}


def load_or_fit_targets(target_path: Path, *, data_root: Path, labels: list[str] | None = None) -> CuboidTargets:
    """Load an existing deterministic fit, or create it through the project fitter.

    Fitting is intentionally delegated to the existing deterministic K-cuboid
    procedure, so target generation is shared rather than copied into a notebook.
    """
    if not target_path.exists():
        if labels is None:
            raise FileNotFoundError(f"Missing {target_path}; supply labels to create it")
        from subprocess import run
        import csv
        manifest = target_path.with_suffix(".labels.csv")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["label"]); writer.writeheader()
            writer.writerows({"label": label} for label in labels)
        script = Path(__file__).parents[1] / "benchmark/scripts/benchmark_k_oriented_cuboids.py"
        run(["python3", str(script), "--data-dir", str(data_root), "--manifest", str(manifest),
             "--output-dir", str(target_path.parent), "--k", str(K)], check=True)
    data = np.load(target_path, allow_pickle=False)
    tokens = tokens_from_components(data["centers"], data["axes"], data["extents"])
    if tokens.ndim != 3 or tokens.shape[1:] != (K, 12):
        raise ValueError(f"Expected (objects, {K}, 12) cuboid targets, got {tokens.shape}")
    return CuboidTargets(data["labels"].astype(str), tokens)
