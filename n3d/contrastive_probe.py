"""Does a trained nonlinear EEG encoder beat ridge? Small participant-aware contrastive model, honest protocol.

Input: binned EEG of one window (default static image 300-500 ms epoch time), z-scored per participant on
training trials. Model: one linear adapter per participant (features -> 256) -> shared 2-layer MLP -> 1024,
InfoNCE (symmetric, learnable temperature) against the objects' CLIP embeddings within each batch.
Augmentation: every training sample averages a random non-empty subset of one participant's trials of an object.
Model selection: best validation CLIP pairwise 2AFC (variant 07); test (08-09) reported once at that epoch.
At test, each participant's repeat-averaged EEG is embedded and the embeddings are averaged across participants.
Null: the same training with training objects' CLIP targets permuted (--perm-seed).

    python -m n3d.contrastive_probe --out runs/probe --seeds 0 1 2
    python -m n3d.contrastive_probe --perm-seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from . import config
from .attr_probe import _zscore_per_sub, full_features, load, pair_2afc, window_features


class Net(nn.Module):
    def __init__(self, d_in, n_sub, hid=256, out=1024, p=0.5):
        super().__init__()
        self.adapt = nn.ModuleList([nn.Linear(d_in, hid) for _ in range(n_sub)])  # out = target dim
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Linear(hid, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, out))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x, sub):
        h = torch.stack([self.adapt[int(s)](xi) for xi, s in zip(x, sub)])
        return self.mlp(h)


def run(a, seed: int, perm_seed: int | None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    b, splits, T = load(a.participants)
    F = full_features(b) if a.stream == "full" else window_features(b, a.stream, a.start, a.width)
    F = _zscore_per_sub(F, b, splits["train"]).astype(np.float32)
    Y = T[a.target].astype(np.float32)
    tr = splits["train"]
    target = {int(o): int(o) for o in tr}
    if perm_seed is not None:
        target = dict(zip(tr.tolist(), np.random.default_rng(1000 + perm_seed).permutation(tr).tolist()))
    subs = np.unique(b.trial_sub)
    groups = {}
    for t in np.flatnonzero(np.isin(b.trial_obj, tr)):
        groups.setdefault((int(b.trial_sub[t]), int(b.trial_obj[t])), []).append(t)
    keys = list(groups)
    sidx = {int(s): i for i, s in enumerate(subs)}
    net = Net(F.shape[1], len(subs), out=Y.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    Yt = torch.from_numpy(Y)

    def embed(objs):
        net.eval()
        with torch.no_grad():
            zs = []
            for s in subs:
                m = b.trial_sub == s
                X = np.stack([F[m & (b.trial_obj == o)].mean(0) for o in objs])
                zs.append(Fnn.normalize(net(torch.from_numpy(X), [sidx[int(s)]] * len(objs)), dim=1))
            return torch.stack(zs).mean(0).numpy()

    best, hist = (-1, None, None), []
    for ep in range(a.epochs):
        net.train()
        order = rng.permutation(len(keys))
        for i in range(0, len(order), a.batch):
            bk = [keys[j] for j in order[i:i + a.batch]]
            # one sample per (participant, object); drop duplicate objects in the batch so InfoNCE negatives are real
            seen, bk2 = set(), []
            for k in bk:
                if k[1] not in seen:
                    seen.add(k[1]); bk2.append(k)
            X = []
            for s, o in bk2:
                ts = np.array(groups[(s, o)])
                pick = ts[rng.random(len(ts)) < 0.5]
                X.append(F[pick if len(pick) else ts[rng.integers(len(ts))][None]].mean(0))
            z = Fnn.normalize(net(torch.from_numpy(np.stack(X)), [sidx[s] for s, _ in bk2]), dim=1)
            c = Fnn.normalize(Yt[[target[o] for _, o in bk2]], dim=1)
            logits = net.logit_scale.exp().clamp(max=100) * z @ c.T
            lab = torch.arange(len(bk2))
            loss = (Fnn.cross_entropy(logits, lab) + Fnn.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % a.eval_every == 0 or ep == a.epochs - 1:
            va = pair_2afc(embed(splits["val"]), Y[splits["val"]])["2afc"]
            hist.append({"epoch": ep, "loss": float(loss), "val_2afc": va})
            if va > best[0]:
                best = (va, ep, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    te = pair_2afc(embed(splits["test"]), Y[splits["test"]], T["_cat"][splits["test"]], T["_color"][splits["test"]])
    return {"seed": seed, "perm_seed": perm_seed, "best_epoch": best[1], "val_2afc": best[0], **{f"test_{k}": v for k, v in te.items()},
            "history": hist}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", nargs="+", default=["sub01", "sub02", "sub03", "sub04", "sub05"])
    ap.add_argument("--stream", default="stat", choices=["stat", "dyn", "full"])
    ap.add_argument("--start", type=int, default=75)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--target", default="clip", choices=["clip", "d2", "sil"], help="embedding the EEG is aligned to")
    ap.add_argument("--perm-seed", type=int, nargs="*", default=None, help="null runs: permuted training targets")
    ap.add_argument("--out", default=str(config.default_runs_root() / "probe"))
    a = ap.parse_args(argv)
    rows = []
    for s in a.seeds:
        for ps in (a.perm_seed if a.perm_seed is not None else [None]):
            r = run(a, s, ps)
            rows.append(r)
            print(f"seed {s} perm {ps}: best epoch {r['best_epoch']} val {r['val_2afc']:.3f} | test 2afc {r['test_2afc']:.3f} "
                  f"within {r['test_2afc_within']:.3f} samecol {r['test_2afc_within_samecol']:.3f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "null" if a.perm_seed is not None else "real"
    (out / f"contrastive_{a.stream}_{a.target}_{tag}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
