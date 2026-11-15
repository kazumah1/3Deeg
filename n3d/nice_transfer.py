"""Does pretraining on THINGS-EEG2 help on EEG-3D? Transfer test with NICE (Song et al., ICLR 2024).

NICE's EEG encoder has a temporal convolution applied to every channel with shared weights
(1 x 25 kernel), then a spatial convolution across all electrodes (63 x 1 kernel). The
spatial layer is tied to THINGS-EEG2's channel order, which EEG-3D does not publish, so it is
always re-initialised here for 64 channels. Everything else can be loaded from the released
checkpoint (huggingface.co/eeyhsong/NICE).

Input: the static stream (64 x 250, 1 s at 250 Hz), z-scored per participant and channel on
training trials, shifted by --shift-ms so stimulus onset lands near t = 0 (EEG-3D's evoked
response suggests onset ~350 ms into the epoch); the end is zero-padded back to 250 samples.

Conditions (each trained --seeds times, same budget):
  scratch          everything randomly initialised
  pretrained       temporal conv (+ projection head) loaded from NICE, all fine-tuned
  pretrained_frozen  NICE temporal conv frozen; spatial layer and heads trained
  random_frozen    random temporal conv frozen (control for pretrained_frozen)
Readout: 72-way category head on the 768-d projection. Checkpoint chosen on validation NLL;
test scored on repeat-averaged EEG per (participant, object), as everywhere else.

    python -m n3d.nice_transfer --run nice_s1_5 --participants sub01 sub02 sub03 sub04 sub05
    python -m n3d.nice_transfer --run nice_shift0 --shift-ms 0      # onset as distributed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .data import load_bundle, make_splits
from .encoder import run_dir
from .probe import averaged, metrics

N_CH, N_T = 64, 250


class Rearrange(nn.Module):  # b e 1 w -> b w e
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class PatchEmbedding(nn.Module):
    """NICE's encoder, identical layer layout so checkpoint keys match (spatial conv excepted)."""

    def __init__(self, n_ch=N_CH, emb=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)), nn.AvgPool2d((1, 51), (1, 5)), nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (n_ch, 1), (1, 1)), nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.5))
        self.projection = nn.Sequential(nn.Conv2d(40, emb, (1, 1)), Rearrange())

    def forward(self, x):
        return self.projection(self.tsconv(x))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class NICE(nn.Module):
    def __init__(self, n_ch=N_CH, n_classes=config.N_CATEGORIES):
        super().__init__()
        self.enc = nn.Sequential(PatchEmbedding(n_ch), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(1440, 768),
                                  ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(768, 768), nn.Dropout(0.5))),
                                  nn.LayerNorm(768))
        self.head = nn.Linear(768, n_classes)

    def forward(self, x):  # (B, 64, 250)
        return self.head(self.proj(self.enc(x[:, None])))

    def temporal_params(self):
        t = self.enc[0].tsconv
        return list(t[0].parameters()) + list(t[2].parameters())


def fetch_weights(local: Path | None) -> tuple[dict, dict]:
    if local:
        enc, proj = local / "pretrain_Enc_eeg_cls.pth", local / "pretrain_Proj_eeg_cls.pth"
    else:
        from huggingface_hub import hf_hub_download
        rev = "7f98b0832c698cb69f7944eaf5462ffa95f204e7"  # pinned: files inspected at this commit
        enc = hf_hub_download("eeyhsong/NICE", "pretrain_Enc_eeg_cls.pth", revision=rev)
        proj = hf_hub_download("eeyhsong/NICE", "pretrain_Proj_eeg_cls.pth", revision=rev)
    def load(p):
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
        except RuntimeError as e:  # the released Enc file is truncated (no zip central directory)
            print(f"torch.load failed on {Path(p).name} ({str(e).splitlines()[0][:60]}...); salvaging intact tensors")
            sd = salvage_truncated(Path(p))
        return {k.replace("module.", ""): v for k, v in sd.items()}
    return load(enc), load(proj)


def salvage_truncated(path: Path) -> dict:
    """Read a torch zip checkpoint that lost its central directory by walking local file headers.

    NICE's pretrain_Enc_eeg_cls.pth on the Hub is 76 kB and cut off inside the spatial conv
    (shallownet.4.weight); every tensor stored before the cut (temporal conv, its BatchNorm,
    projection) is complete. Tensors whose bytes are missing or short are dropped."""
    import io
    import pickle
    import re

    b = path.read_bytes()
    descs = [m.start() for m in re.finditer(rb"PK\x07\x08", b)]
    ents = {}
    for m in re.finditer(rb"PK\x03\x04", b):
        i = m.start()
        nlen, xlen = (int.from_bytes(b[i + o:i + o + 2], "little") for o in (26, 28))
        name, s = b[i + 30:i + 30 + nlen].decode(), i + 30 + nlen + xlen
        ents[name.split("/", 1)[1]] = b[s:next((d for d in descs if d > s), len(b))]

    def rebuild(storage, offset, size, stride, *_):
        dtype, raw = storage
        n = int(np.prod(size)) if len(size) else 1
        el = torch.tensor([], dtype=dtype).element_size()
        if raw is None or len(raw) < (offset + n) * el:
            return None
        flat = torch.frombuffer(bytearray(raw), dtype=dtype)
        return flat.as_strided(tuple(size), tuple(stride), offset).clone()

    class U(pickle.Unpickler):
        def persistent_load(self, pid):  # ('storage', FloatStorage, key, location, numel)
            return pid[1].dtype, ents.get(f"data/{pid[2]}")

        def find_class(self, mod, name):  # allow-list: a state_dict needs only these
            if name.startswith("_rebuild_tensor"):
                return rebuild
            if (mod, name) == ("collections", "OrderedDict") or (mod == "torch" and name.endswith("Storage")):
                return super().find_class(mod, name)
            raise pickle.UnpicklingError(f"refusing {mod}.{name} in checkpoint")

    sd = U(io.BytesIO(ents["data.pkl"])).load()
    dropped = [k for k, v in sd.items() if v is None]
    if dropped:
        print(f"  dropped (bytes missing): {dropped}")
    return {k: v for k, v in sd.items() if v is not None}


def load_pretrained(model: NICE, enc_sd: dict, proj_sd: dict, load_proj: bool) -> list[str]:
    """Copy every tensor whose name and shape match; the spatial conv (63 vs 64 channels) never
    matches. Names in NICE's file look like '0.tsconv.0.weight'; ours are 'enc.0.tsconv.0.weight'."""
    own = model.state_dict()
    loaded = []
    # the released file names the block 'shallownet' (e.g. '0.shallownet.0.weight'); same layout as tsconv
    enc_sd = {k.replace("shallownet.", "tsconv."): v for k, v in enc_sd.items()}
    for k, v in enc_sd.items():  # match by the part after 'tsconv.' / 'projection.', whatever the prefix
        for tag in ("tsconv.", "projection."):
            if tag in k:
                key = "enc.0." + k[k.index(tag):]
                # tsconv.4 is the spatial conv (63 ch, never matches), tsconv.5 its BatchNorm: keep fresh
                if key in own and own[key].shape == v.shape and not key.startswith(("enc.0.tsconv.4.", "enc.0.tsconv.5.")):
                    own[key] = v; loaded.append(key)
    if load_proj:
        for k, v in proj_sd.items():
            key = "proj." + k
            if key in own and own[key].shape == v.shape:
                own[key] = v; loaded.append(key)
    model.load_state_dict(own)
    if not any("tsconv.0.weight" in k for k in loaded):
        raise SystemExit("temporal conv weights not found in the NICE checkpoint; keys were:\n  "
                         + "\n  ".join(f"{k} {tuple(v.shape)}" for k, v in enc_sd.items()))
    return loaded


def prepare(b, s, shift_ms: float) -> np.ndarray:
    X = b.stat.astype(np.float32).copy()
    tr = np.isin(b.trial_obj, s["train"])
    for sub in np.unique(b.trial_sub):
        m = b.trial_sub == sub
        mu = X[m & tr].mean((0, 2), keepdims=True)
        sd = X[m & tr].std((0, 2), keepdims=True) + 1e-6
        X[m] = (X[m] - mu) / sd
    k = round(shift_ms * 250 / 1000)
    if k:
        X = np.concatenate([X[:, :, k:], np.zeros_like(X[:, :, :k])], 2)
    return X


def run_condition(cond, seed, X, b, s, args, dev, weights):
    torch.manual_seed(seed); np.random.seed(seed)
    model = NICE().to(dev)
    note = []
    if cond.startswith("pretrained"):
        note = load_pretrained(model, *weights, load_proj=not args.no_proj)
    frozen = cond.endswith("_frozen")
    if frozen:
        for p in model.temporal_params():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    tr = np.flatnonzero(np.isin(b.trial_obj, s["train"]))
    Xtr = torch.from_numpy(X[tr]).to(dev); ytr = torch.from_numpy(b.obj_cat[b.trial_obj[tr]]).to(dev)
    Xva, yva, _, _ = averaged(X, b, s["val"]); Xte, yte, ste, _ = averaged(X, b, s["test"])
    K = config.N_CATEGORIES

    @torch.no_grad()
    def probs(Z):
        model.eval()
        return torch.softmax(model(torch.from_numpy(Z).to(dev)), 1).cpu().numpy()

    best, log = None, []
    for ep in range(args.epochs):
        model.train()
        if frozen:  # keep frozen BN statistics fixed too
            model.enc[0].tsconv[2].eval()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), args.batch_size):
            j = perm[i:i + args.batch_size]
            x = Xtr[j] + args.noise * torch.randn_like(Xtr[j])
            loss = F.cross_entropy(model(x), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        mv = metrics(probs(Xva), yva, K)
        log.append({"epoch": ep, "loss": float(loss), "val_nll": mv["nll"], "val_top5": mv["top5"]})
        if best is None or mv["nll"] < best[0]:
            best = (mv["nll"], ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    P = probs(Xte)
    res = {"condition": cond, "seed": seed, "best_epoch": best[1], "val_nll": best[0],
           "test": metrics(P, yte, K),
           "test_per_subject": {args.participants[u]: metrics(P[ste == u], yte[ste == u], K) for u in np.unique(ste)},
           "loaded": len(note), "log": log}
    return res, P, yte


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--split", default="ours", choices=list(config.SPLITS))
    ap.add_argument("--shift-ms", type=float, default=350)
    ap.add_argument("--conditions", nargs="+", default=["scratch", "pretrained", "pretrained_frozen", "random_frozen"],
                    choices=["scratch", "pretrained", "pretrained_frozen", "random_frozen"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--noise", type=float, default=0.1, help="Gaussian input noise (z units) during training")
    ap.add_argument("--no-proj", action="store_true", help="load only the temporal conv, not NICE's projection head")
    ap.add_argument("--weights-dir", type=Path, default=None, help="local folder with pretrain_*_cls.pth (default: download)")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = load_bundle(args.data_root, tuple(args.participants), "point_cloud_simple")
    s = make_splits(b, args.split)
    X = prepare(b, s, args.shift_ms)
    weights = fetch_weights(args.weights_dir) if any(c.startswith("pretrained") for c in args.conditions) else None
    if weights:
        print("NICE checkpoint tensors:", {k: tuple(v.shape) for k, v in weights[0].items()})
    out, ens = [], {}
    for cond in args.conditions:
        for seed in args.seeds:
            r, P, y = run_condition(cond, seed, X, b, s, args, dev, weights)
            out.append(r); ens.setdefault(cond, []).append(P)
            t = r["test"]
            print(f"{cond:18s} seed {seed}: best ep {r['best_epoch']:2d} val_nll {r['val_nll']:.3f} | "
                  f"test top1 {t['top1']:.3f} top5 {t['top5']:.3f} nll {t['nll']:.3f} | loaded {r['loaded']} tensors")
    summary = {}
    for cond, Ps in ens.items():
        rows = [r["test"] for r in out if r["condition"] == cond]
        summary[cond] = {k: [float(np.mean([m[k] for m in rows])), float(np.std([m[k] for m in rows]))]
                         for k in ("top1", "top5", "nll")}
        summary[cond]["ensemble"] = metrics(np.mean(Ps, 0), y, config.N_CATEGORIES)
    (run_dir(args.run) / "nice_transfer.json").write_text(json.dumps(
        {"summary": summary, "runs": out, "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    print(f"\n{'condition':18s} {'top1 mean±sd':>16s} {'top5 mean±sd':>16s} {'nll mean±sd':>16s} {'ens top5':>9s} {'ens nll':>8s}")
    for cond, m in summary.items():
        f = lambda k: f"{m[k][0]:.3f}±{m[k][1]:.3f}"
        print(f"{cond:18s} {f('top1'):>16s} {f('top5'):>16s} {f('nll'):>16s} {m['ensemble']['top5']:9.3f} {m['ensemble']['nll']:8.3f}")
    print(f"chance: top1 {1/72:.3f} top5 {5/72:.3f} nll {np.log(72):.3f}; wrote {run_dir(args.run) / 'nice_transfer.json'}")


if __name__ == "__main__":
    main()
