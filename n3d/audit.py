"""Encoder replication audit: how much of the paper's EEG classification number is protocol?

Per participant and seed, two encoder runs:
  ours      train 00-06, select best.pt on val 07, report test 08-09 (honest protocol)
  upstream  train 00-07, no val, select on test 08-09 (the paper's protocol)

Every epoch logs test metrics, so each run also yields the "oracle" number: the best test
score over epochs, i.e. what selecting on test buys. Summary rows (mean ± sd over
participants x seeds):
  ours@val        test score at the val-selected epoch            <- what we should report
  ours@test-max   best test score over epochs, same run           <- selection bias alone
  upstream@test   best test score over epochs, paper's split      <- paper protocol (+1 train variant)

    python -m n3d.audit --print                       # list commands
    modal run scripts/modal_n3d.py::audit             # one GPU job per participant
    python -m n3d.audit --summarize                   # table from runs under N3D_RUNS
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from . import config
from .sweep import Job

PARTICIPANTS = ("sub01", "sub02", "sub03", "sub04", "sub05")
SEEDS = (0, 1)
SPLITS = ("ours", "upstream")
METRICS = ("shape_top1", "shape_top5", "shape_nll")
CHANCE = {"shape_top1": 1 / 72, "shape_top5": 5 / 72, "shape_nll": float(np.log(72))}


def run_name(prefix: str, p: str, split: str, seed: int) -> str:
    return f"{prefix}_{p}_{split}_s{seed}"


def jobs(participants=PARTICIPANTS, seeds=SEEDS, epochs: int = 200, prefix: str = "audit",
         null: bool = False) -> list[Job]:
    """null=True: permutation null (--shuffle-train), ours split only. Its ours@test-max row is what
    picking the best test epoch yields with no EEG-shape signal at all."""
    out, splits, extra = [], (("ours",) if null else SPLITS), (["--shuffle-train"] if null else [])
    for p in participants:
        cmds = [["python", "-m", "n3d.encoder", "train", "--run", run_name(prefix, p, split, seed),
                 "--participants", p, "--split", split, "--seed", str(seed), "--epochs", str(epochs),
                 "--keep-snapshots", "1", *extra]
                for split in splits for seed in seeds]
        out.append(Job(f"{prefix}_{p}", "enc", cmds))
    return out


def _records(d: Path) -> list[dict]:
    f = d / "metrics.jsonl"
    if not f.exists():
        return []
    return [r for r in map(json.loads, f.read_text().splitlines()) if "test_shape_top1" in r]


def per_run(d: Path, split: str) -> dict | None:
    recs = _records(d)
    if not recs:
        return None
    out = {"epochs": len(recs)}
    for m in METRICS:
        test = np.array([r[f"test_{m}"] for r in recs])
        best = np.argmin(test) if m.endswith("nll") else np.argmax(test)
        out[f"test_max_{m}"] = float(test[best])
        if split == "ours":  # same selection rule as best.pt (val shape_nll, lower is better)
            sel = int(np.argmin([r["val_shape_nll"] for r in recs]))
            out[f"at_val_{m}"] = float(test[sel])
            out["val_epoch"] = recs[sel]["epoch"]
    return out


def summarize(root: Path, prefix: str = "audit") -> str:
    rows = {"ours@val": [], "ours@test-max": [], "upstream@test": []}
    per = []
    for d in sorted(root.glob(f"{prefix}_sub*_*_s*")):
        split = "upstream" if "_upstream_" in d.name else "ours"
        r = per_run(d, split)
        if r is None:
            continue
        per.append({"run": d.name, **r})
        if split == "ours":
            rows["ours@val"].append({m: r[f"at_val_{m}"] for m in METRICS})
            rows["ours@test-max"].append({m: r[f"test_max_{m}"] for m in METRICS})
        else:
            rows["upstream@test"].append({m: r[f"test_max_{m}"] for m in METRICS})
    lines = ["| protocol | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for name, rs in rows.items():
        if not rs:
            continue
        cells = [f"{np.mean([r[m] for r in rs]):.4f}±{np.std([r[m] for r in rs]):.4f}" for m in METRICS]
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    lines.append("| chance | – | " + " | ".join(f"{CHANCE[m]:.4f}" for m in METRICS) + " |")
    (root / f"{prefix}_summary.json").write_text(json.dumps({"rows": rows, "per_run": per}, indent=1))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", nargs="+", default=list(PARTICIPANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--prefix", default="audit")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--null", action="store_true", help="permutation-null runs (--shuffle-train)")
    a = ap.parse_args(argv)
    if a.summarize:
        print(summarize(config.default_runs_root(), a.prefix))
        return
    for j in jobs(tuple(a.participants), tuple(a.seeds), a.epochs, a.prefix, a.null):
        print(f"# {j.name}")
        for c in j.cmds:
            print(" ", shlex.join(c))


if __name__ == "__main__":
    main()
