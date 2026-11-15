"""Cuboid tokens, rotation conversion, and deterministic surface rendering."""
from __future__ import annotations

import itertools
import numpy as np


K = 3
TOKEN_DIM = 12


def rotation_6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the first two *columns* of a 3x3 rotation as a 6D vector."""
    return np.asarray(matrix, dtype=np.float32)[..., :, :2].reshape(*matrix.shape[:-2], 6)


def matrix_from_rotation_6d(value):
    """Continuous 6D rotation conversion (Zhou et al.), for NumPy or Torch."""
    if hasattr(value, "new_zeros"):
        import torch
        first, second = value[..., :3], value[..., 3:]
        first = torch.nn.functional.normalize(first, dim=-1)
        second = second - (first * second).sum(-1, keepdim=True) * first
        second = torch.nn.functional.normalize(second, dim=-1)
        third = torch.cross(first, second, dim=-1)
        return torch.stack((first, second, third), dim=-1)
    first, second = np.asarray(value)[..., :3], np.asarray(value)[..., 3:]
    first = first / np.clip(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8, None)
    second = second - (first * second).sum(-1, keepdims=True) * first
    second = second / np.clip(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8, None)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def tokens_from_components(centers: np.ndarray, axes: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """Pack K oriented cuboids into center/log-half-extent/6D-rotation tokens."""
    return np.concatenate((centers, np.log(np.maximum(extents / 2, 1e-6)), rotation_6d_from_matrix(axes)), axis=-1).astype(np.float32)


def component_sample_counts(log_half_extents: np.ndarray, total: int) -> np.ndarray:
    extents = np.exp(log_half_extents) * 2
    areas = 2 * (extents[..., 0] * extents[..., 1] + extents[..., 0] * extents[..., 2] + extents[..., 1] * extents[..., 2])
    raw = areas / np.maximum(areas.sum(), 1e-8) * total
    counts = np.floor(raw).astype(int)
    counts[np.argsort(raw - counts)[-(total - counts.sum()):]] += 1
    return counts


def cuboid_surface_samples(tokens: np.ndarray, total: int = 2048, seed: int = 0) -> np.ndarray:
    """Sample the union of oriented-cuboid surfaces without changing component order."""
    rng = np.random.default_rng(seed)
    counts = component_sample_counts(tokens[:, 3:6], total)
    clouds = []
    for token, count in zip(tokens, counts):
        center, half, axes = token[:3], np.exp(token[3:6]), matrix_from_rotation_6d(token[6:])
        areas = np.array([half[1]*half[2], half[1]*half[2], half[0]*half[2], half[0]*half[2], half[0]*half[1], half[0]*half[1]])
        faces = rng.choice(6, count, p=areas / areas.sum())
        local = rng.uniform(-half, half, (count, 3))
        for axis in range(3):
            local[faces == 2*axis, axis] = -half[axis]
            local[faces == 2*axis + 1, axis] = half[axis]
        clouds.append(center + local @ axes.T)
    return np.concatenate(clouds, axis=0).astype(np.float32)


def cuboid_symmetries() -> tuple[np.ndarray, ...]:
    """All 24 proper signed permutations; cuboid parameterization equivalences."""
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float32)
            matrix[permutation, np.arange(3)] = signs
            if np.linalg.det(matrix) > 0:
                matrices.append(matrix)
    return tuple(matrices)
