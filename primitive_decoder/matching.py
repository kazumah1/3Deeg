"""Permutation- and cuboid-symmetry-aware matching losses."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import cuboid_symmetries, matrix_from_rotation_6d


def _rotation_angle(first, second):
    import torch
    relative = first.transpose(-1, -2) @ second
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine)


def pair_cost(predicted, target):
    """KxK detached assignment cost, minimized over proper cuboid symmetries."""
    import torch
    p_rotation = matrix_from_rotation_6d(predicted[:, 6:])
    pieces = []
    for symmetry in cuboid_symmetries():
        s = torch.as_tensor(symmetry, dtype=target.dtype, device=target.device)
        # Columns are permuted/sign-flipped; extents follow the same columns.
        order = torch.argmax(s.abs(), dim=0)
        target_rot = matrix_from_rotation_6d(target[:, 6:]) @ s
        target_extent = target[:, 3:6][:, order]
        center = (predicted[:, None, :3] - target[None, :, :3]).square().mean(-1)
        extent = (predicted[:, None, 3:6] - target_extent[None]).square().mean(-1)
        angle = _rotation_angle(p_rotation[:, None], target_rot[None])
        pieces.append(center + extent + angle.square())
    return torch.stack(pieces).amin(0)


def hungarian_indices(predicted, target):
    """Return target component indices matched to each predicted component."""
    cost = pair_cost(predicted, target).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(len(rows), dtype=np.int64); order[rows] = cols
    return order


def matched_parameter_loss(predicted, target):
    """Mean center/log-half-extent/rotation loss after Hungarian assignment.

    Symmetries are applied while constructing the assignment cost. The final
    rotation term again takes the minimum over equivalent cuboid axes.
    """
    import torch
    center_losses, extent_losses, rotation_losses = [], [], []
    for one_pred, one_target in zip(predicted, target):
        matched = one_target[torch.as_tensor(hungarian_indices(one_pred, one_target), device=one_target.device)]
        center_losses.append((one_pred[:, :3] - matched[:, :3]).square().mean())
        best = []
        p_rotation = matrix_from_rotation_6d(one_pred[:, 6:])
        for symmetry in cuboid_symmetries():
            s = torch.as_tensor(symmetry, dtype=one_target.dtype, device=one_target.device)
            order = torch.argmax(s.abs(), dim=0)
            r = matrix_from_rotation_6d(matched[:, 6:]) @ s
            e = matched[:, 3:6][:, order]
            best.append((one_pred[:, 3:6] - e).square().mean(-1) + _rotation_angle(p_rotation, r).square())
        best = torch.stack(best).amin(0)
        extent_losses.append(best.mean())
        rotation_losses.append(torch.zeros((), device=one_pred.device))
    center = torch.stack(center_losses).mean(); extent_rotation = torch.stack(extent_losses).mean()
    return {"total": center + extent_rotation, "center": center, "extent_rotation": extent_rotation}
