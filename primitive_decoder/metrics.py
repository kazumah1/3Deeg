"""Point-cloud metrics shared by primitive and point-cloud decoder studies."""
from __future__ import annotations

import numpy as np


def symmetric_squared_chamfer(prediction: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    first = cKDTree(target).query(prediction, workers=-1)[0]
    second = cKDTree(prediction).query(target, workers=-1)[0]
    return float(np.mean(first**2) + np.mean(second**2))


def normalized_chamfer(prediction: np.ndarray, target: np.ndarray) -> float:
    diagonal_squared = float(np.square(target.max(0) - target.min(0)).sum())
    return symmetric_squared_chamfer(prediction, target) / max(diagonal_squared, 1e-8)


def f_score(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.02) -> float:
    from scipy.spatial import cKDTree
    precision = np.mean(cKDTree(target).query(prediction, workers=-1)[0] <= threshold)
    recall = np.mean(cKDTree(prediction).query(target, workers=-1)[0] <= threshold)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def geometry_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.02) -> dict[str, float]:
    return {"chamfer_squared": symmetric_squared_chamfer(prediction, target),
            "chamfer_normalized": normalized_chamfer(prediction, target),
            f"fscore@{threshold:g}": f_score(prediction, target, threshold)}
