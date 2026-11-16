"""Noise predictors for point-cloud diffusion: eps = f(x_t, t, cond).

pvcnn     upstream PVCNN2 as Neuro-3D uses it. CUDA only.
pointnet  a small pure-PyTorch PointNet denoiser (per-point residual MLP + global max-pool).
          Runs on CPU, so the whole pipeline can be smoke-tested anywhere; also the
          Branch B "is PVN necessary?" comparison point.

Conditioning (--cond-mode):
  concat  pvcnn: raw 1024-d condition concatenated to every point's XYZ (upstream design).
          pointnet: condition projected to `width` and concatenated at the input layer.
  film    the condition predicts a per-channel scale and shift (zero-initialised, so the
          model starts unconditional).
          pvcnn: applied to the output features of every SA and FP stage via forward
          hooks (the upstream module is imported unchanged; input is XYZ only).
          pointnet: applied before every hidden layer, from cond + timestep embedding.
  xattn   pointnet only: the condition is split into `n_tokens` chunks, each projected by one
          shared linear layer plus a learned position, and every residual block cross-attends
          from points to those tokens (zero-initialised output).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

COND_MODES = {"pvcnn": ("concat", "film"), "pointnet": ("concat", "film", "xattn")}


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class PVCNNDenoiser(nn.Module):
    def __init__(self, cond_dim: int = 1024, embed_dim: int = 64, dropout: float = 0.1,
                 width_multiplier: float = 1, voxel_resolution_multiplier: float = 1, cond_mode: str = "concat"):
        super().__init__()
        from .upstream import pvcnn2_class
        self.cond_mode = cond_mode
        extra = cond_dim if cond_mode == "concat" else 0
        self.net = pvcnn2_class()(num_classes=3, embed_dim=embed_dim, use_att=True, dropout=dropout,
                                  extra_feature_channels=extra, width_multiplier=width_multiplier,
                                  voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.net.classifier[-1].bias.data.normal_(0, 1e-6)
        self.net.classifier[-1].weight.data.normal_(0, 1e-6)
        self.film = None
        if cond_mode == "film":
            stages = list(self.net.sa_layers) + list(self.net.fp_layers)
            chans = self._stage_channels(stages)
            self.film = nn.ModuleList([nn.Linear(cond_dim, 2 * c) for c in chans])
            for f in self.film:
                nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
            self._gb = None
            for i, s in enumerate(stages):
                s.register_forward_hook(self._make_hook(i))

    @torch.no_grad()
    def _stage_channels(self, stages) -> list[int]:
        """Output feature channels of each stage, read off one dummy forward (CUDA ops need a GPU)."""
        chans, hooks = {}, []
        for i, s in enumerate(stages):
            hooks.append(s.register_forward_hook(lambda m, inp, out, i=i: chans.__setitem__(i, out[0].shape[1])))
        dev = next(self.net.parameters()).device
        self.net.cuda().eval()
        self.net(torch.randn(1, 3, 2048, device="cuda"), torch.zeros(1, device="cuda", dtype=torch.long))
        self.net.to(dev).train()
        for h in hooks:
            h.remove()
        return [chans[i] for i in range(len(stages))]

    def _make_hook(self, i):
        def hook(module, inputs, out):
            if self._gb is None:
                return out
            gamma, beta = self._gb[i]
            return (out[0] * (1 + gamma[:, :, None]) + beta[:, :, None],) + tuple(out[1:])
        return hook

    def forward(self, x_t, t, cond):
        if self.cond_mode == "concat":
            inp = torch.cat([x_t, cond[:, None, :].expand(-1, x_t.shape[1], -1)], dim=2)
        else:
            inp = x_t
            self._gb = [f(cond).chunk(2, dim=1) for f in self.film]
        try:
            with torch.autocast("cuda", dtype=torch.float32):  # upstream PVNModel forces fp32
                return self.net(inp.transpose(1, 2), t).transpose(1, 2)
        finally:
            self._gb = None


class CrossAttn(nn.Module):
    def __init__(self, width: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, h, tokens):
        return self.attn(self.norm(h), tokens, tokens, need_weights=False)[0]


class PointNetDenoiser(nn.Module):
    def __init__(self, cond_dim: int = 1024, width: int = 256, depth: int = 4, cond_mode: str = "concat",
                 t_dim: int = 64, n_tokens: int = 16):
        super().__init__()
        self.cond_mode, self.t_dim, self.n_tokens, self.width = cond_mode, t_dim, n_tokens, width
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, width), nn.SiLU(), nn.Linear(width, width))
        if cond_mode == "xattn":  # cond split into n_tokens chunks, one shared projection (cheap in params)
            assert cond_dim % n_tokens == 0
            self.c_proj = nn.Linear(cond_dim // n_tokens, width)
            self.tok_pos = nn.Parameter(torch.randn(n_tokens, width) * 0.02)
            self.xattn = nn.ModuleList([CrossAttn(width) for _ in range(depth)])
        else:
            self.c_proj = nn.Linear(cond_dim, width)
            self.xattn = None
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
        if self.xattn is not None:
            tokens = self.c_proj(cond.view(B, self.n_tokens, -1)) + self.tok_pos
        else:
            c = self.c_proj(cond)
        parts = [x_t, te[:, None].expand(-1, N, -1)]
        if self.cond_mode == "concat":
            parts.append(c[:, None].expand(-1, N, -1))
        h = self.inp(torch.cat(parts, dim=2))
        for i, (lin, norm) in enumerate(zip(self.local, self.norms)):
            if self.xattn is not None:
                h = h + self.xattn[i](h, tokens)
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
    if name not in COND_MODES:
        raise ValueError(name)
    if cond_mode not in COND_MODES[name]:
        raise ValueError(f"{name} supports cond_mode in {COND_MODES[name]}, not {cond_mode!r}")
    cls = PVCNNDenoiser if name == "pvcnn" else PointNetDenoiser
    return cls(cond_dim=cond_dim, cond_mode=cond_mode, **kw)


def pointnet_params(cond_mode: str, width: int, depth: int = 4, cond_dim: int = 1024, n_tokens: int = 16) -> int:
    return sum(p.numel() for p in PointNetDenoiser(cond_dim, width, depth, cond_mode, n_tokens=n_tokens).parameters())


def match_width(cond_mode: str, target: int, depth: int = 4, cond_dim: int = 1024, n_tokens: int = 16) -> int:
    """Smallest multiple of 8 whose PointNet denoiser has >= target parameters (for size-matched grids)."""
    w = 32
    while pointnet_params(cond_mode, w, depth, cond_dim, n_tokens) < target:
        w += 8
    return w
