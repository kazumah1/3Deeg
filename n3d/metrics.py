"""Point-set distances. One convention for the whole project.

Primary score (same as benchmark/ in this repo):
    CD = mean_a min_b ||a-b||^2 + mean_b min_a ||b-a||^2        (symmetric, squared, mean)
Also reported, because papers mix conventions:
    CD_L1 = mean_a min_b ||a-b|| + mean_b min_a ||b-a||          (unsquared)
    F-score@tau: harmonic mean of precision (pred within tau of gt) and recall (gt within tau of pred)
Both clouds must already be in the same frame (we use upstream pc_norm: centred, unit max radius).
Neuro-3D does not ship its evaluation code, so its 5.35e-2 figure cannot be mapped onto these
numbers until its convention is known; compare models only inside this harness.
"""
from __future__ import annotations

import numpy as np
import torch

TAUS = (0.01, 0.02, 0.05)


def _summary(d_ab: np.ndarray, d_ba: np.ndarray, taus=TAUS) -> dict:
    out = {
        "cd": float((d_ab ** 2).mean() + (d_ba ** 2).mean()),
        "cd_acc": float((d_ab ** 2).mean()),   # pred -> gt: stray predicted points
        "cd_cov": float((d_ba ** 2).mean()),   # gt -> pred: missing object parts
        "cd_l1": float(d_ab.mean() + d_ba.mean()),
    }
    for t in taus:
        p, r = float((d_ab < t).mean()), float((d_ba < t).mean())
        out[f"f@{t}"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return out


def chamfer_np(pred: np.ndarray, gt: np.ndarray, taus=TAUS) -> dict:
    """Exact, CPU, via KD-trees. pred (N,3), gt (M,3)."""
    from scipy.spatial import cKDTree
    d_ab, _ = cKDTree(gt).query(pred, k=1)
    d_ba, _ = cKDTree(pred).query(gt, k=1)
    return _summary(d_ab, d_ba, taus)


@torch.no_grad()
def nn_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """For each point of a (B,N,3), distance to the nearest point of b (B,M,3)."""
    out = []
    for i in range(0, a.shape[1], chunk):
        out.append(torch.cdist(a[:, i:i + chunk].float(), b.float()).amin(dim=2))
    return torch.cat(out, dim=1)


def chamfer_torch(pred: torch.Tensor, gt: torch.Tensor, taus=TAUS) -> list[dict]:
    """Batched, exact, runs on whatever device the tensors are on. Returns one dict per item."""
    d_ab, d_ba = nn_dist(pred, gt).cpu().numpy(), nn_dist(gt, pred).cpu().numpy()
    return [_summary(d_ab[i], d_ba[i], taus) for i in range(len(d_ab))]


def chamfer_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Differentiable squared symmetric Chamfer, mean over batch. Use for regression baselines."""
    d = torch.cdist(pred, gt) ** 2
    return d.amin(dim=2).mean(dim=1).add(d.amin(dim=1).mean(dim=1)).mean()
