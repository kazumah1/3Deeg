"""Summarize the EEG -> CLIP-decoder bridge (modal_n3d.py::bridge): one row per decoder x norm,
mean ± sd over participants, next to the same decoder fed true CLIP (the ceiling).

    python -m n3d.bridge --decoders bench_clip_pn_concat_s0 bench_clip_pn_film_s0 bench_clip_pv_film_s0
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config

KEYS = [("cd", "CD"), ("cd_best", "CD best-of-k"), ("f02", "F@0.02"), ("inst_top1", "inst top1"),
        ("cat_top1", "cat top1"), ("afc_within", "2AFC within-cat")]


def _row(d: Path, suffix: str) -> dict | None:
    ev, idf = d / f"eval_test_{suffix}.json", d / f"identify_pred_test_{suffix}.json"
    if not ev.exists():
        return None
    s = json.loads(ev.read_text())["summary"]
    r = {"cd": s["cd_mean_over_samples"], "cd_best": s["cd_best_of_k"], "f02": s["f@0.02_mean"]}
    if idf.exists():
        i = json.loads(idf.read_text())
        r |= {"inst_top1": i["inst_top1"], "cat_top1": i["cat_top1"], "afc_within": i["afc_within_cat"]}
    return r


def summarize(root: Path, decoders, base="k5_s100") -> str:
    groups = defaultdict(list)
    for dec in decoders:
        d = root / dec
        if r := _row(d, base):
            groups[(dec, "true cond (ceiling)")].append(r)
        for ev in sorted(d.glob(f"eval_test_{base}_*.json")):
            m = re.match(rf"eval_test_{base}_(eeg|ctg|cf)_(.+)_(match|none|topk|sample|top1|nn1|nn5)\.json", ev.name)
            if m and (r := _row(d, f"{base}_{m[1]}_{m[2]}_{m[3]}")):
                kind = "cf_" + re.sub(r"_(sub\d+|perm\d+)$", r"_\1", m[2]).replace("clippred_", "") if m[1] == "cf" else m[1]
                kind = re.sub(r"_sub\d+$", "_subject", re.sub(r"_perm\d+$", "_perm", kind))
                groups[(dec, f"{kind}/{m[3]}")].append(r)
    lines = ["| decoder | cond | n | " + " | ".join(h for _, h in KEYS) + " |", "|" + "---|" * (len(KEYS) + 3)]
    for (dec, cond), rs in groups.items():
        cells = []
        for k, _ in KEYS:
            v = [r[k] for r in rs if k in r]
            cells.append("–" if not v else f"{np.mean(v):.4f}" + (f"±{np.std(v):.4f}" if len(v) > 1 else ""))
        lines.append(f"| {dec} | {cond} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoders", nargs="+",
                    default=["bench_clip_pn_concat_s0", "bench_clip_pn_film_s0", "bench_clip_pv_film_s0",
                             "bench_category_pn_film_s0", "bench_category_pn_concat_s0"])
    ap.add_argument("--base", default="k5_s100")
    a = ap.parse_args(argv)
    print(summarize(config.default_runs_root(), a.decoders, a.base))


if __name__ == "__main__":
    main()
