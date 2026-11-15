"""Object-disjoint splits used by the cuboid decoder benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def _variant(label: str) -> int:
    return int(label.rsplit("_", 1)[1])


def make_object_splits(labels: list[str] | tuple[str, ...]) -> Split:
    """Split variants 00-06/07/08-09 into train/validation/test.

    The original paper uses 00-07 for training and 08-09 for testing.  This
    benchmark reserves variant 07 for validation so every object identity is
    in exactly one split while retaining the paper's held-out test objects.
    """
    labels = tuple(sorted(set(map(str, labels))))
    train = tuple(x for x in labels if _variant(x) <= 6)
    validation = tuple(x for x in labels if _variant(x) == 7)
    test = tuple(x for x in labels if _variant(x) >= 8)
    if not train or not validation or not test:
        raise ValueError("Expected variants 00-09 for train, validation, and test splits")
    verify_no_leakage(Split(train, validation, test))
    return Split(train, validation, test)


def verify_no_leakage(split: Split, benchmark_manifest: Path | None = None) -> None:
    groups = (set(split.train), set(split.validation), set(split.test))
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Object identity leakage between splits")
    if benchmark_manifest is not None:
        with benchmark_manifest.open(newline="", encoding="utf-8") as stream:
            benchmark = {row["label"] for row in csv.DictReader(stream)}
        leaked = benchmark & set(split.train)
        if leaked:
            raise ValueError(f"Fixed primitive benchmark leaked into training: {sorted(leaked)}")
