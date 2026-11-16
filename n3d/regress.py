"""One-shot regression decoder: condition -> point cloud, trained with Chamfer loss.

The control for "does the generator family matter?". A deterministic decoder trained on
Chamfer is known to drift toward the conditional mean shape when the condition is weak;
this measures how far that goes on our data, under the same splits, condition sources and
metrics as n3d.diffusion. Output files match `n3d.diffusion sample`, so n3d.identify,
n3d.parity and n3d.aggregate read them unchanged.

Decoder: MLP 1024 -> hidden -> hidden -> N*3 (Achlioptas et al. 2018 style).
Checkpoint chosen on validation Chamfer.

    python -m n3d.regress train --run rg_clip --cond-source clip --participants sub01
    python -m n3d.regress sample --run rg_clip            # writes pred_test_k1_s0.npz + eval json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .data import EEG3D
from .diffusion import Conditioner, setup
from .encoder import device, load_trained, run_dir, seed_worker, set_seed
from .metrics import chamfer_loss, chamfer_torch


class MLPDecoder(nn.Module):
    def __init__(self, cond_dim=1024, hidden=1024, n_points=2048):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, n_points * 3))

    def forward(self, cond):
        return self.net(cond).view(-1, self.n_points, 3)


def build(args, dev, b):
    enc = load_trained(args.encoder_run, dev) if args.cond_source == "eeg" else None
    cond = Conditioner(args.cond_source, enc, b.clip_video).to(dev)
    return cond, MLPDecoder(1024, args.hidden, args.n_points).to(dev)


@torch.no_grad()
def predict(cond, net, b, objs, dev, bs=48):
    net.eval()
    preds, gts, os_ = [], [], []
    for batch in DataLoader(EEG3D(b, objs, train=False), batch_size=bs):
        preds.append(net(cond(batch, dev)).cpu()); gts.append(batch["points"]); os_.append(torch.as_tensor(batch["obj"]))
    return torch.cat(preds), torch.cat(gts), torch.cat(os_)


def train(args):
    set_seed(args.seed)
    dev, out = device(), run_dir(args.run)
    b, splits = setup(args)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    ds = EEG3D(b, splits["train"], train=True, seed=args.seed)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, drop_last=True,
                        num_workers=2, persistent_workers=True, worker_init_fn=seed_worker)
    cond, net = build(args, dev, b)
    params = [p for p in list(net.parameters()) + list(cond.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in net.parameters())
    (out / "model.json").write_text(json.dumps({"denoiser_params": n_params}))
    print(f"regression decoder: {n_params/1e6:.2f}M params; cond={args.cond_source}; points {args.n_points}")
    eval_split = "val" if len(splits.get("val", [])) else "test"
    log, step, best, t0 = out / "metrics.jsonl", 0, None, time.time()
    while step < args.max_steps:
        for batch in loader:
            net.train()
            x0 = batch["points"].to(dev)
            x0 = x0[:, torch.randperm(x0.shape[1], device=dev)[:args.n_points]]
            c = cond(batch, dev)
            if torch.rand(()) > 0.5:
                c = c + torch.randn_like(c) * 0.004
            loss = chamfer_loss(net(c), x0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 50.0); opt.step(); step += 1
            if step % args.eval_every == 0 or step == args.max_steps:
                P, G, _ = predict(cond, net, b, splits[eval_split], dev)
                cd = float(np.mean([m["cd"] for m in chamfer_torch(P.to(dev), G.to(dev))]))
                rec = {"step": step, "loss": float(loss), f"{eval_split}_cd": cd, "sec": time.time() - t0}
                print(rec); open(log, "a").write(json.dumps(rec) + "\n")
                if best is None or cd < best[0]:
                    best = (cd, step)
                    torch.save({"net": net.state_dict(), "cond": cond.state_dict(), "step": step}, out / "best.pt")
            if step >= args.max_steps:
                break
    print(f"done; best {eval_split} cd {best[0]:.4g} at step {best[1]}")


def cmd_sample(args):
    out = run_dir(args.run)
    saved = json.loads((out / "config.json").read_text())
    for k in ("cond_source", "encoder_run", "geometry", "participants", "split", "tiny", "hidden", "n_points"):
        setattr(args, k, saved[k])
    dev = device()
    b, splits = setup(args)
    cond, net = build(args, dev, b)
    ck = torch.load(out / "best.pt", map_location=dev, weights_only=True)
    net.load_state_dict(ck["net"]); cond.load_state_dict(ck["cond"], strict=False)
    P, G, O = predict(cond, net, b, splits[args.eval_split], dev)
    if args.cond_override == "zeros":
        with torch.no_grad():
            P = net(torch.zeros(len(O), 1024, device=dev)).cpu()
    tag = "_zeros" if args.cond_override == "zeros" else ""
    np.savez_compressed(out / f"pred_{args.eval_split}_k1_s0{tag}.npz", pred=P[:, None].numpy(), gt=G.numpy(), obj=O.numpy())
    rows = [{"obj": o, "sample": 0, **m} for o, m in zip(O.tolist(), chamfer_torch(P.to(dev), G.to(dev)))]
    cd = np.array([r["cd"] for r in rows])
    summary = {"split": args.eval_split, "n_objects": int(len(O)), "samples": 1, "steps": 0,
               "cd_mean_over_samples": float(cd.mean()), "cd_best_of_k": float(cd.mean()),
               "f@0.02_mean": float(np.mean([r["f@0.02"] for r in rows])), "tag": tag, "best_step": ck["step"]}
    (out / f"eval_{args.eval_split}_k1_s0{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--encoder-run", default=None)
    t.add_argument("--cond-source", default="clip", choices=["eeg", "clip", "category"])
    t.add_argument("--data-root", type=Path, default=None)
    t.add_argument("--participants", nargs="+", default=["sub01"])
    t.add_argument("--geometry", default="point_cloud_simple")
    t.add_argument("--split", default="ours", choices=list(config.SPLITS))
    t.add_argument("--tiny", action="store_true")
    t.add_argument("--hidden", type=int, default=1024)
    t.add_argument("--n-points", type=int, default=2048)
    t.add_argument("--max-steps", type=int, default=20_000)
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=1234)
    s = sub.add_parser("sample")
    s.add_argument("--run", required=True)
    s.add_argument("--data-root", type=Path, default=None)
    s.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    s.add_argument("--cond-override", default="none", choices=["none", "zeros"])
    args = ap.parse_args(argv)
    {"train": train, "sample": cmd_sample}[args.cmd](args)


if __name__ == "__main__":
    main()
