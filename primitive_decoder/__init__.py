"""Reusable components for the frozen-EEG-to-cuboid benchmark."""

from .splits import Split, make_object_splits, verify_no_leakage
from .targets import CuboidTargets, load_or_fit_targets

__all__ = ["Split", "CuboidTargets", "load_or_fit_targets", "make_object_splits", "verify_no_leakage"]
