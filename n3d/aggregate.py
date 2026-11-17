"""Collect decoder-benchmark results into one table (mean ± sd over seeds).

Reads every run directory under N3D_RUNS whose name starts with --prefix and has a
config.json: eval_test_*.json (Chamfer, F-score), swap_test_*.json (does the output follow
the condition?), identify_*.json (instance / category identification), model.json (params).
Baseline rows come from a baselines run (n3d.baselines) if given.

    python -m n3d.aggregate --prefix bench_clip --baselines baselines_sub01
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config

COLS = [("params_M", "params"), ("cd", "CD"), ("cd_best", "CD best-of-k"), ("f02", "F@0.02"),
        ("swap_own", "swap own"), ("swap_zero", "swap zeros"), ("follow_other", "follows other-cat"),
        ("inst_top1", "inst top1"), ("cat_top1", "cat top1"), ("afc_within", "2AFC within-cat")]


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect(root: Path, prefix: str, tag: str | None = None) -> list[dict]:
    """tag (e.g. "k5_s100") pins which sampling run to read; default: the last eval file found."""
    rows = []
    for d in sorted(root.glob(f"{prefix}*")):
        if not (d / "config.json").exists():
            continue
        r = {"run": d.name, "group": re.sub(r"_s\d+$", "", d.name)}
        if m := _load(d / "model.json"):
            r["params_M"] = m["denoiser_params"] / 1e6
        ev = sorted(d.glob(f"eval_test_{tag}.json" if tag else "eval_test_k*_s*.json"))
        ev = [e for e in ev if not re.search(r"_(zeros|cat\w+)\.json$", e.name)]
        if ev:
            s = _load(ev[-1])["summary"]
            r |= {"cd": s["cd_mean_over_samples"], "cd_best": s["cd_best_of_k"], "f02": s["f@0.02_mean"]}
        if sw := next(iter(sorted(d.glob("swap_test_*.json"))), None):
            s = _load(sw)
            r |= {"swap_own": s["own"]["cd_to_own_gt"], "swap_zero": s["zeros"]["cd_to_own_gt"],
                  "follow_other": s["other_category"].get("follows_donor_frac")}
        if idf := next(iter(sorted(d.glob(f"identify_pred_test_{tag}.json" if tag else "identify_pred_test_*.json"))), None):
            s = _load(idf)
            r |= {"inst_top1": s["inst_top1"], "cat_top1": s["cat_top1"], "afc_within": s["afc_within_cat"]}
        rows.append(r)
    return rows


def table(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    head = "| model | n | " + " | ".join(h for _, h in COLS) + " |"
    lines = [head, "|" + "---|" * (len(COLS) + 2)]
    for g, rs in sorted(groups.items(), key=lambda kv: np.mean([r.get("cd", np.inf) for r in kv[1]])):
        cells = []
        for k, _ in COLS:
            v = [r[k] for r in rs if r.get(k) is not None]
            if not v:
                cells.append("–")
            elif k == "params_M":
                cells.append(f"{np.mean(v):.2f}")
            else:
                cells.append(f"{np.mean(v):.4f}" + (f"±{np.std(v):.4f}" if len(v) > 1 else ""))
        lines.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="bench")
    ap.add_argument("--baselines", default=None, help="n3d.baselines run name, for reference rows")
    ap.add_argument("--out", type=Path, default=None, help="also write markdown here")
    ap.add_argument("--tag", default=None, help="sampling run to read, e.g. k5_s100 or k5_s1000")
    a = ap.parse_args(argv)
    root = config.default_runs_root()
    rows = collect(root, a.prefix, a.tag)
    if a.tag:
        rows = [r for r in rows if "cd" in r]
    md = table(rows)
    if a.baselines and (b := _load(root / a.baselines / "baselines_summary.json")):
        md += "\n\nBaselines (test, mean CD):\n" + "\n".join(
            f"- {k}: {v['mean']:.4f}" for k, v in sorted(((k, v) for k, v in b.items() if isinstance(v, dict)), key=lambda kv: kv[1]['mean']))
    print(md)
    if a.out:
        a.out.write_text(md + "\n")
    (root / f"aggregate_{a.prefix}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
