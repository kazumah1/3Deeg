"""Decoder benchmark: which generator works, separately from whether the EEG carries the signal.

Tiers (run in order; each answers a different question):
  clip      condition = the object's CLIP video feature (the encoder's training target), so
            decoders are compared as if the encoder were perfect.
  category  condition = a learned embedding of the true category only: how much of a good
            output is just "the right category's typical shape".
  eeg       condition = frozen EEG encoder output (needs --encoder-run). Only worth running
            for decoders that clearly beat the category-template baseline under `clip`.

Grid in every tier (same data, splits, point count, steps and seeds):
  pv_concat  PVCNN2, condition concatenated to every point (Neuro-3D)       CUDA
  pv_film    PVCNN2, FiLM on every SA/FP stage, XYZ-only input               CUDA
  pn_concat  PointNet, size-matched, concat at input
  pn_film    PointNet, size-matched, FiLM every block
  pn_xattn   PointNet, size-matched, cross-attention to 16 condition tokens
  rg         one-shot MLP regression on Chamfer (mean-shape collapse control)

Each job: train -> sample test (k samples, DDIM) -> swap test -> identification.

    python -m n3d.sweep --tier clip --print            # list the shell commands
    python -m n3d.sweep --tier clip --run-local pn_film_s0   # run one job here (CPU ok for pn/rg)
    modal run scripts/modal_n3d.py::sweep --tier clip  # all jobs in parallel on Modal GPUs
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

PN_TARGET_PARAMS = 1_400_000  # PointNet FiLM at width 256; concat/xattn widths are matched to it

MODELS = {
    "pv_concat": ("diffusion", ["--denoiser", "pvcnn", "--cond-mode", "concat"]),
    "pv_film": ("diffusion", ["--denoiser", "pvcnn", "--cond-mode", "film"]),
    "pn_concat": ("diffusion", ["--denoiser", "pointnet", "--cond-mode", "concat", "--pn-match-params", str(PN_TARGET_PARAMS)]),
    "pn_film": ("diffusion", ["--denoiser", "pointnet", "--cond-mode", "film", "--pn-match-params", str(PN_TARGET_PARAMS)]),
    "pn_xattn": ("diffusion", ["--denoiser", "pointnet", "--cond-mode", "xattn", "--pn-match-params", str(PN_TARGET_PARAMS)]),
    "rg": ("regress", []),
}


@dataclass
class Job:
    name: str
    model: str
    cmds: list[list[str]] = field(default_factory=list)

    @property
    def needs_cuda(self) -> bool:
        return self.model.startswith("pv")


def jobs(tier: str, seeds=(0, 1), steps: int = 30_000, n_points: int = 8192, samples: int = 5,
         sample_steps: int = 100, participants=("sub01",), encoder_run: str | None = None,
         models=tuple(MODELS), prefix: str = "bench") -> list[Job]:
    if tier == "eeg" and not encoder_run:
        raise SystemExit("tier eeg needs --encoder-run")
    out = []
    for m in models:
        kind, extra = MODELS[m]
        for seed in seeds:
            run = f"{prefix}_{tier}_{m}_s{seed}"
            common = ["--participants", *participants, "--cond-source", tier, "--seed", str(seed)]
            if encoder_run:
                common += ["--encoder-run", encoder_run]
            py = ["python", "-m"]  # runners substitute their own interpreter
            if kind == "diffusion":
                train = [*py, "n3d.diffusion", "train", "--run", run, *common, *extra, "--max-steps", str(steps),
                         "--n-points", str(n_points), "--eval-every", str(max(steps // 10, 50))]
                pred = f"pred_test_k{samples}_s{sample_steps}.npz"
                cmds = [train,
                        [*py, "n3d.diffusion", "sample", "--run", run, "--samples", str(samples), "--steps", str(sample_steps)],
                        [*py, "n3d.diffusion", "swap", "--run", run, "--steps", str(sample_steps)]]
            else:
                # one-shot Chamfer on 8192 points needs a B x 8192 x 8192 cdist (12 GB at batch 48): cap at 2048
                train = [*py, "n3d.regress", "train", "--run", run, *common, "--max-steps", str(steps // 2),
                         "--n-points", str(min(n_points, 2048))]
                pred = "pred_test_k1_s0.npz"
                cmds = [train, [*py, "n3d.regress", "sample", "--run", run]]
            cmds.append([*py, "n3d.identify", "--run", run, "--file", pred])
            out.append(Job(run, m, cmds))
    return out


def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--tier", required=True, choices=["clip", "category", "eeg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--n-points", type=int, default=8192,
                    help="PVCNN costs about the same at 2048 and 8192 (voxel ops dominate), so default to the GT count")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sample-steps", type=int, default=100, help="DDIM steps when sampling")
    ap.add_argument("--participants", nargs="+", default=["sub01"])
    ap.add_argument("--encoder-run", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--prefix", default="bench")


def from_args(a) -> list[Job]:
    return jobs(a.tier, tuple(a.seeds), a.steps, a.n_points, a.samples, a.sample_steps, participants=tuple(a.participants),
                encoder_run=a.encoder_run, models=tuple(a.models), prefix=a.prefix)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--run-local", nargs="*", default=None, help="job names to run here, in order")
    a = ap.parse_args(argv)
    js = from_args(a)
    if a.run_local is None or a.print:
        for j in js:
            print(f"# {j.name}{'  [CUDA]' if j.needs_cuda else ''}")
            for c in j.cmds:
                print(" ", shlex.join(c))
        return
    for j in js:
        if a.run_local and j.name not in a.run_local:
            continue
        for c in j.cmds:
            c = [sys.executable, *c[1:]]
            print("+", shlex.join(c), flush=True)
            subprocess.run(c, check=True)


if __name__ == "__main__":
    main()
