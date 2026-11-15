"""Decoder baselines.  They deliberately consume one pooled f_g vector only."""
from __future__ import annotations


def _torch():
    import torch
    from torch import nn
    return torch, nn


class LinearCuboidDecoder:
    def __new__(cls, embedding_dim: int = 1024, components: int = 3):
        _, nn = _torch()
        return nn.Sequential(nn.Linear(embedding_dim, components * 12), nn.Unflatten(-1, (components, 12)))


class MLPCuboidDecoder:
    def __new__(cls, embedding_dim: int = 1024, components: int = 3, width: int = 512):
        _, nn = _torch()
        return nn.Sequential(nn.Linear(embedding_dim, width), nn.GELU(), nn.LayerNorm(width),
                             nn.Linear(width, width), nn.GELU(), nn.Linear(width, components * 12),
                             nn.Unflatten(-1, (components, 12)))
