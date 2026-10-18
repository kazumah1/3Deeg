"""Notebook helpers: training curves and side-by-side point clouds (plotly, renders in Colab)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .encoder import run_dir


def curves(run: str, keys: list[str], x: str | None = None):
    import plotly.graph_objects as go
    rows = [json.loads(l) for l in (run_dir(run) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    x = x or ("epoch" if "epoch" in rows[0] else "step")
    fig = go.Figure()
    for k in keys:
        pts = [(r[x], r[k]) for r in rows if k in r]
        if pts:
            fig.add_scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], name=k, mode="lines+markers")
    fig.update_layout(title=run, xaxis_title=x, height=380)
    return fig


def clouds(pred: np.ndarray, gt: np.ndarray, title: str = "", n: int = 4096):
    """Generated (left) and ground truth (right), same normalised frame."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("generated", "ground truth"))
    rng = np.random.default_rng(0)
    for col, pts in ((1, pred), (2, gt)):
        p = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
        fig.add_scatter3d(x=p[:, 0], y=p[:, 2], z=p[:, 1], mode="markers",
                          marker=dict(size=1.5, color=p[:, 1], colorscale="Viridis"), row=1, col=col)
    for s in ("scene", "scene2"):
        fig.layout[s].update(aspectmode="cube", xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]),
                             zaxis=dict(range=[-1, 1]))
    fig.update_layout(title=title, height=420, showlegend=False)
    return fig


def show_samples(run: str, file: str | None = None, k: int = 3):
    """Plot the first k objects from the newest samples_*.npz (training) or a pred_*.npz file."""
    d = run_dir(run)
    path = d / file if file else sorted(d.glob("samples_step*.npz"))[-1]
    z = np.load(path)
    pred = z["pred"] if z["pred"].ndim == 3 else z["pred"][:, 0]
    figs = [clouds(pred[i], z["gt"][i], f"{path.name} | obj {int(z['obj'][i])}") for i in range(min(k, len(pred)))]
    for f in figs:
        f.show()
