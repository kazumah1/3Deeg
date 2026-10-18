"""Noise predictors for point-cloud diffusion: eps = f(x_t, t, cond).

pvcnn     upstream PVCNN2 exactly as Neuro-3D uses it: the condition vector is
          concatenated to every point's XYZ (in_channels = 3 + 1024). CUDA only.
pointnet  a small pure-PyTorch PointNet denoiser (per-point MLP + global max-pool).
          Runs on CPU, so the whole pipeline can be smoke-tested anywhere; also the
          first Branch B "is PVN necessary?" comparison point.
Conditioning (pointnet only): 'concat' like upstream, or 'film' where the
condition predicts a per-channel scale and shift for every hidden layer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class PVCNNDenoiser(nn.Module):
    def __init__(self, cond_dim: int = 1024, embed_dim: int = 64, dropout: float = 0.1,
                 width_multiplier: float = 1, voxel_resolution_multiplier: float = 1):
        super().__init__()
        from .upstream import pvcnn2_class
        self.net = pvcnn2_class()(num_classes=3, embed_dim=embed_dim, use_att=True, dropout=dropout,
                                  extra_feature_channels=cond_dim, width_multiplier=width_multiplier,
                                  voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.net.classifier[-1].bias.data.normal_(0, 1e-6)
        self.net.classifier[-1].weight.data.normal_(0, 1e-6)

    def forward(self, x_t, t, cond):
        inp = torch.cat([x_t, cond[:, None, :].expand(-1, x_t.shape[1], -1)], dim=2)
        with torch.autocast("cuda", dtype=torch.float32):  # upstream PVNModel forces fp32
            return self.net(inp.transpose(1, 2), t).transpose(1, 2)


class PointNetDenoiser(nn.Module):
    def __init__(self, cond_dim: int = 1024, width: int = 256, depth: int = 4, cond_mode: str = "concat",
                 t_dim: int = 64):
        super().__init__()
        self.cond_mode, self.t_dim = cond_mode, t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.c_proj = nn.Linear(cond_dim, width)
        in_dim = 3 + width + (width if cond_mode == "concat" else 0)
        self.inp = nn.Linear(in_dim, width)
        self.local = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.film = nn.ModuleList([nn.Linear(width, 2 * width) for _ in range(depth)]) if cond_mode == "film" else None
        self.mix = nn.Linear(2 * width, width)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)
        if self.film is not None:
            for f in self.film:
                nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)

    def forward(self, x_t, t, cond):
        B, N, _ = x_t.shape
        te = self.t_mlp(timestep_embedding(t, self.t_dim))
        c = self.c_proj(cond)
        parts = [x_t, te[:, None].expand(-1, N, -1)]
        if self.cond_mode == "concat":
            parts.append(c[:, None].expand(-1, N, -1))
        h = self.inp(torch.cat(parts, dim=2))
        for i, (lin, norm) in enumerate(zip(self.local, self.norms)):
            z = norm(h)
            if self.film is not None:
                gamma, beta = self.film[i](c + te).chunk(2, dim=1)
                z = z * (1 + gamma[:, None]) + beta[:, None]
            h = h + lin(nn.functional.silu(z))
            if i == len(self.local) // 2 - 1:  # one global exchange mid-way
                g = h.amax(dim=1, keepdim=True).expand(-1, N, -1)
                h = self.mix(torch.cat([h, g], dim=2))
        return self.out(h)


def build_denoiser(name: str, cond_dim: int = 1024, cond_mode: str = "concat", **kw) -> nn.Module:
    if name == "pvcnn":
        if cond_mode != "concat":
            raise ValueError("pvcnn supports cond_mode=concat only (upstream design); use pointnet for film")
        return PVCNNDenoiser(cond_dim=cond_dim, **kw)
    if name == "pointnet":
        return PointNetDenoiser(cond_dim=cond_dim, cond_mode=cond_mode, **kw)
    raise ValueError(name)
