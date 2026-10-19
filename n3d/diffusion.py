"""Conditional point-cloud diffusion (Neuro-3D shape stage) plus failure-point tests.

Faithful to recon_main.py / ds_shape_color_generation.py where it matters:
  DDPM, 1000 steps, linear betas 1e-5 -> 8e-3, epsilon prediction, MSE loss;
  encoder frozen (loaded from an `n3d.encoder` run); condition = f_g (1024-d, the
  geometry head) concatenated to each point; with p=0.5 Gaussian noise (std 0.004)
  added to the condition during training; AdamW lr 1e-3 betas (0.95, 0.999),
  wd 1e-6 (none on bias/LayerNorm), cosine schedule with 2000 warm-up steps,
  grad-norm clip 50, batch 48.
Added for diagnosis:
  --cond-source eeg | clip | category
      eeg      : the real pipeline
      clip     : condition on the object's CLIP video feature (the encoder's training
                 target) = what the decoder could do with a perfect encoder
      category : condition on a learned category embedding only = how much a
                 recognisable output owes to category alone
  `swap` command: same initial noise, own vs other objects' conditions.

    python -m n3d.diffusion train --run dm_tiny --encoder-run enc_tiny --tiny --max-steps 3000
    python -m n3d.diffusion train --run dm_sub01 --encoder-run enc_sub01
    python -m n3d.diffusion sample --run dm_sub01 --split test --samples 5 --steps 100
    python -m n3d.diffusion swap --run dm_sub01 --steps 100
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .data import EEG3D, load_bundle, make_splits
from .denoisers import build_denoiser
from .ckpt import atomic_save, load_resume, log_line, save_resume, save_snapshot
from .encoder import device, load_trained, restore_rng, rng_state, run_dir, seed_worker, set_seed
from .metrics import chamfer_torch


def make_schedulers():
    from diffusers import DDIMScheduler, DDPMScheduler
    kw = dict(num_train_timesteps=1000, beta_start=1e-5, beta_end=8e-3, beta_schedule="linear", clip_sample=False)
    return DDPMScheduler(**kw), DDIMScheduler(**kw)


class Conditioner(nn.Module):
    """Maps a batch to the 1024-d condition vector according to --cond-source."""

    def __init__(self, source: str, encoder: nn.Module | None, clip_table: np.ndarray):
        super().__init__()
        self.source, self.encoder = source, encoder
        self.register_buffer("clip_table", torch.from_numpy(clip_table).float(), persistent=False)
        self.cat_emb = nn.Embedding(config.N_CATEGORIES, 1024) if source == "category" else None
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad_(False)

    def forward(self, batch, dev) -> torch.Tensor:
        if self.source == "eeg":
            self.encoder.eval()
            with torch.no_grad():
                f_g, *_ = self.encoder(batch["dyn"].to(dev).float(), batch["stat"].to(dev).float())
            return f_g
        if self.source == "clip":
            return self.clip_table[torch.as_tensor(batch["obj"], device=dev)]
        return self.cat_emb(torch.as_tensor(batch["cat"], device=dev))


def build(args, dev, b):
    enc = load_trained(args.encoder_run, dev) if args.cond_source == "eeg" else None
    cond = Conditioner(args.cond_source, enc, b.clip_video).to(dev)
    kw = {}
    if args.denoiser == "pvcnn":
        kw = dict(width_multiplier=args.width, voxel_resolution_multiplier=args.voxel_res)
    net = build_denoiser(args.denoiser, cond_dim=1024, cond_mode=args.cond_mode, **kw).to(dev)
    return cond, net


def trainable_params(cond, net):
    no_decay = ("bias", "LayerNorm.weight", "norm")
    named = [(n, p) for n, p in list(net.named_parameters()) + list(cond.named_parameters()) if p.requires_grad]
    return [{"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": 1e-6},
            {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0}]


def cosine_with_warmup(opt, warmup, total):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def sample(net, cond_vec, sched, n_points=config.N_POINTS, steps=100, generator=None, x_T=None):
    """DDIM when steps < 1000 (eta=0, deterministic given x_T), else full DDPM."""
    B, dev = cond_vec.shape[0], cond_vec.device
    x = x_T if x_T is not None else torch.randn(B, n_points, 3, device=dev, generator=generator)
    sched.set_timesteps(steps)
    for t in sched.timesteps:
        tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
        eps = net(x, tt, cond_vec)
        x = sched.step(eps, t, x).prev_sample
    return x


def ckpt_state(net, cond, opt, sched, step, extra=None) -> dict:
    return {"net": net.state_dict(), "cond": {k: v for k, v in cond.state_dict().items() if not k.startswith("encoder.")},
            "opt": opt.state_dict() if opt else None, "sched": sched.state_dict() if sched else None,
            "step": step, "rng": rng_state(), **(extra or {})}


def load_ckpt(path_or_ck, net, cond, dev, opt=None, sched=None):
    ck = path_or_ck if isinstance(path_or_ck, dict) else torch.load(path_or_ck, map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"])
    cond.load_state_dict(ck["cond"], strict=False)
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if sched is not None and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    return ck


def setup(args):
    b = load_bundle(args.data_root, tuple(args.participants), args.geometry)
    cats = [0, 14, 26, 51] if args.tiny else None
    splits = make_splits(b, args.split, categories=cats)
    if args.tiny:  # overfit check: train and evaluate on the same few training objects
        splits = {"train": splits["train"][::4][:8]} | {k: splits["train"][::4][:8] for k in ("val", "test")}
    return b, splits


@torch.no_grad()
def evaluate(net, cond, b, objs, dev, steps, n_eval, seed=0, ddim=None):
    ds = EEG3D(b, objs[:n_eval], train=False)
    batch = next(iter(DataLoader(ds, batch_size=len(ds))))
    g = torch.Generator(device=dev).manual_seed(seed)
    pred = sample(net, cond(batch, dev), ddim, steps=steps, generator=g)
    m = chamfer_torch(pred, batch["points"].to(dev))
    return {k: float(np.mean([x[k] for x in m])) for k in m[0]}, pred.cpu().numpy(), batch


def train(args):
    set_seed(args.seed)
    dev, out = device(), run_dir(args.run)
    b, splits = setup(args)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    ds = EEG3D(b, splits["train"], train=True, seed=args.seed)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, drop_last=True,
                        num_workers=2, persistent_workers=True, worker_init_fn=seed_worker)
    cond, net = build(args, dev, b)
    ddpm, ddim = make_schedulers()
    opt = torch.optim.AdamW(trainable_params(cond, net), lr=args.lr, betas=(0.95, 0.999))
    lrs = cosine_with_warmup(opt, min(2000, args.max_steps // 10), args.max_steps)
    step = 0
    ck = None if args.restart else load_resume(out, dev)
    if ck is not None:
        step = load_ckpt(ck, net, cond, dev, opt, lrs)["step"]
        restore_rng(ck)
        print(f"resumed at step {step}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"denoiser {args.denoiser}/{args.cond_mode}: {n_params/1e6:.2f}M params; cond={args.cond_source}; "
          f"train objects {len(splits['train'])}, samples {len(ds)}")
    eval_split = "val" if len(splits.get("val", [])) else "test"
    log = out / "metrics.jsonl"
    alphas = ddpm.alphas_cumprod.to(dev)
    t0, running = time.time(), []
    while step < args.max_steps:
        for batch in loader:
            net.train()
            x0 = batch["points"].to(dev)
            c = cond(batch, dev)
            if torch.rand(()) > 0.5:
                c = c + torch.randn_like(c) * 0.004
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.shape[0],), device=dev)
            a = alphas[t][:, None, None]
            x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
            loss = nn.functional.mse_loss(net(x_t, t, c), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 50.0)
            opt.step(); lrs.step(); step += 1
            running.append(loss.item())
            if step % args.log_every == 0:
                rec = {"step": step, "loss": float(np.mean(running)), "grad_norm": float(gn),
                       "lr": lrs.get_last_lr()[0], "sec_per_step": (time.time() - t0) / args.log_every}
                running, t0 = [], time.time()
                if step % args.eval_every == 0 or step == args.max_steps:
                    m, pred, batch_e = evaluate(net, cond, b, splits[eval_split], dev, args.eval_steps, args.n_eval, ddim=ddim)
                    rec |= {f"{eval_split}_{k}": v for k, v in m.items()}
                    np.savez_compressed(out / f"samples_step{step:07d}.npz", pred=pred,
                                        gt=batch_e["points"].numpy(), obj=np.asarray(batch_e["obj"]))
                    t0 = time.time()
                print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
                log_line(log, rec)
            if step % args.ckpt_every == 0 or step == args.max_steps:
                state = ckpt_state(net, cond, opt, lrs, step)
                save_resume(state, out)
                if step % args.snapshot_every == 0 or step == args.max_steps:
                    save_snapshot(state, out, f"step{step:07d}.pt", keep=args.keep_snapshots)
                if step % args.milestone_every == 0:  # weights only, never pruned: for evaluating along training
                    (out / "milestones").mkdir(exist_ok=True)
                    atomic_save(ckpt_state(net, cond, None, None, step), out / "milestones" / f"step{step:07d}.pt")
            if step >= args.max_steps:
                break
    print(f"done at step {step}; checkpoints in {out}")


def load_for_inference(args):
    dev = device()
    saved = json.loads((run_dir(args.run) / "config.json").read_text())
    for k in ("denoiser", "cond_mode", "cond_source", "encoder_run", "width", "voxel_res", "geometry",
              "participants", "split", "tiny"):
        setattr(args, k, saved[k])
    b, splits = setup(args)
    cond, net = build(args, dev, b)
    load_ckpt(run_dir(args.run) / args.ckpt, net, cond, dev)
    net.eval()
    return dev, b, splits, cond, net


def cmd_sample(args):
    dev, b, splits, cond, net = load_for_inference(args)
    ddpm, ddim = make_schedulers()
    sched = ddpm if args.steps >= 1000 else ddim
    ds = EEG3D(b, splits[args.eval_split], train=False)
    preds, gts, objs = [], [], []
    for batch in DataLoader(ds, batch_size=args.batch_size):
        c = cond(batch, dev)
        per = []
        for k in range(args.samples):
            g = torch.Generator(device=dev).manual_seed(1000 + k)
            per.append(sample(net, c, sched, steps=args.steps, generator=g).cpu())
        preds.append(torch.stack(per, 1)); gts.append(batch["points"]); objs.append(torch.as_tensor(batch["obj"]))
    P, G, O = torch.cat(preds), torch.cat(gts), torch.cat(objs)
    rows = []
    for k in range(args.samples):
        for o, m in zip(O.tolist(), chamfer_torch(P[:, k].to(dev), G.to(dev))):
            rows.append({"obj": o, "sample": k, **m})
    cd = np.array([r["cd"] for r in rows]).reshape(args.samples, -1)
    summary = {"split": args.eval_split, "n_objects": int(len(O)), "samples": args.samples, "steps": args.steps,
               "cd_mean_over_samples": float(cd.mean()), "cd_best_of_k": float(cd.min(0).mean()),
               "f@0.02_mean": float(np.mean([r["f@0.02"] for r in rows]))}
    out = run_dir(args.run)
    np.savez_compressed(out / f"pred_{args.eval_split}_k{args.samples}_s{args.steps}.npz", pred=P.numpy(), gt=G.numpy(), obj=O.numpy())
    (out / f"eval_{args.eval_split}_k{args.samples}_s{args.steps}.json").write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary, indent=1))


def cmd_swap(args):
    """Does the decoder use the condition, and does it move toward the right object?

    For each evaluation object i with fixed initial noise, generate from:
      own condition, a same-category sibling's, a different-category object's, zeros,
    then score every output against i's ground truth and against the donor's.
    If outputs follow the donor (CD to donor < CD to own) the condition is used;
    if 'zeros' is as good as 'own', the decoder ignores it.
    """
    dev, b, splits, cond, net = load_for_inference(args)
    _, ddim = make_schedulers()
    ds = EEG3D(b, splits[args.eval_split], train=False)
    batch = next(iter(DataLoader(ds, batch_size=len(ds))))
    C = cond(batch, dev)
    objs, cats = np.asarray(batch["obj"]), np.asarray(batch["cat"])
    rng = np.random.default_rng(0)
    G = batch["points"].to(dev)
    xT = torch.randn(len(objs), config.N_POINTS, 3, device=dev, generator=torch.Generator(device=dev).manual_seed(7))
    donors = {"own": np.arange(len(objs))}
    donors["sibling"] = np.array([rng.choice(np.flatnonzero((cats == c) & (objs != o))) if ((cats == c) & (objs != o)).any() else i
                                  for i, (o, c) in enumerate(zip(objs, cats))])
    donors["other_category"] = np.array([rng.choice(np.flatnonzero(cats != c)) for c in cats])
    res = {}
    for name, d in list(donors.items()) + [("zeros", None)]:
        c = torch.zeros_like(C) if d is None else C[torch.as_tensor(d, device=dev)]
        out = torch.cat([sample(net, c[i:i + args.batch_size], ddim, steps=args.steps, x_T=xT[i:i + args.batch_size])
                         for i in range(0, len(objs), args.batch_size)])
        to_own = np.mean([m["cd"] for m in chamfer_torch(out, G)])
        rec = {"cd_to_own_gt": float(to_own)}
        if d is not None and name != "own":
            rec["cd_to_donor_gt"] = float(np.mean([m["cd"] for m in chamfer_torch(out, G[torch.as_tensor(d, device=dev)])]))
            rec["follows_donor_frac"] = float(np.mean([a["cd"] < c_["cd"] for a, c_ in
                                                       zip(chamfer_torch(out, G[torch.as_tensor(d, device=dev)]), chamfer_torch(out, G))]))
        res[name] = rec
        print(name, rec)
    (run_dir(args.run) / f"swap_{args.eval_split}_s{args.steps}.json").write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--encoder-run", default=None, help="n3d.encoder run providing best.pt (cond-source eeg)")
    t.add_argument("--cond-source", default="eeg", choices=["eeg", "clip", "category"])
    t.add_argument("--denoiser", default="pvcnn", choices=["pvcnn", "pointnet"])
    t.add_argument("--cond-mode", default="concat", choices=["concat", "film"])
    t.add_argument("--width", type=float, default=1.0, help="pvcnn width multiplier (Branch B)")
    t.add_argument("--voxel-res", type=float, default=1.0, help="pvcnn voxel resolution multiplier (Branch B)")
    t.add_argument("--data-root", type=Path, default=None)
    t.add_argument("--participants", nargs="+", default=["sub01"])
    t.add_argument("--geometry", default="point_cloud_simple")
    t.add_argument("--split", default="ours", choices=list(config.SPLITS))
    t.add_argument("--tiny", action="store_true")
    t.add_argument("--max-steps", type=int, default=100_000, help="upstream stops at 100,500")
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--eval-steps", type=int, default=50, help="DDIM steps for in-training eval")
    t.add_argument("--n-eval", type=int, default=16)
    t.add_argument("--ckpt-every", type=int, default=1000, help="steps between rolling last.pt saves")
    t.add_argument("--snapshot-every", type=int, default=5000, help="steps between frozen full snapshots")
    t.add_argument("--keep-snapshots", type=int, default=3, help="full snapshots kept (they include optimizer state)")
    t.add_argument("--milestone-every", type=int, default=10000, help="steps between weights-only milestones")
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--restart", action="store_true")
    for name in ("sample", "swap"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--ckpt", default="last.pt")
        p.add_argument("--data-root", type=Path, default=None)
        p.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
        p.add_argument("--steps", type=int, default=100, help="DDIM steps; >=1000 uses full DDPM")
        p.add_argument("--samples", type=int, default=5)
        p.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args(argv)
    if args.cmd == "train" and args.cond_source == "eeg" and not args.encoder_run:
        ap.error("--encoder-run is required with --cond-source eeg")
    {"train": train, "sample": cmd_sample, "swap": cmd_swap}[args.cmd](args)


if __name__ == "__main__":
    main()
