"""Run n3d jobs on Modal GPUs: the decoder benchmark sweep, single commands, and data upload.

One-time setup (from the repo root, with `pip install modal` and `modal token new` done):
    modal volume create n3d-data
    modal volume put n3d-data ~/3deeg-data/neuro-3d/packed neuro-3d/packed

Then:
    modal run scripts/modal_n3d.py::smoke                         # PVCNN build + 200 steps, prints sec/step
    modal run scripts/modal_n3d.py::sweep --tier clip --dry-run   # list jobs and GPUs, launch nothing
    modal run --detach scripts/modal_n3d.py::sweep --tier clip    # launch every job in parallel
    modal run scripts/modal_n3d.py::cmd --args "python -m n3d.nice_transfer --run nice_s1_5 --participants sub01"
    modal run scripts/modal_n3d.py::aggregate --prefix bench_clip
    modal volume get n3d-runs bench_clip_pn_film_s0 ./runs/          # pull a run back

Volumes: n3d-data (packed arrays, read-mostly), n3d-runs (checkpoints and results; jobs resume
from last.pt if preempted), n3d-cache (upstream clone + compiled PVCNN CUDA ops, built once for
the GPU architectures listed in TORCH_CUDA_ARCH_LIST).
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

app = modal.App("n3d")
data = modal.Volume.from_name("n3d-data", create_if_missing=True)
runs = modal.Volume.from_name("n3d-runs", create_if_missing=True)
cache = modal.Volume.from_name("n3d-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.6.0", "numpy<2.3", "scipy", "pandas", "openpyxl", "scikit-learn", "einops>=0.8",
                 "diffusers>=0.27,<0.37", "ninja>=1.11", "huggingface_hub")
    .env({"N3D_DATA": "/data/neuro-3d", "N3D_RUNS": "/runs", "N3D_UPSTREAM": "/cache/upstream",
          "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions",
          "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("n3d")
)
VOLS = {"/data": data, "/runs": runs, "/cache": cache}
DEFAULT_GPU = "L4"


@app.function(image=image, gpu=DEFAULT_GPU, volumes=VOLS, timeout=24 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_job(name: str, cmds: list[list[str]]) -> dict:
    """Run one job's commands in order; training resumes from last.pt after a preemption."""
    import time
    log = Path("/runs") / name / "modal.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), []
    for c in cmds:
        c = [sys.executable, *c[1:]] if c[0] == "python" else c
        with open(log, "a") as f:
            f.write(f"+ {shlex.join(c)}\n"); f.flush()
            r = subprocess.run(c, stdout=f, stderr=subprocess.STDOUT)
        runs.commit(); cache.commit()
        done.append({"cmd": c[2:4], "rc": r.returncode})
        if r.returncode:
            tail = log.read_text().splitlines()[-25:]
            return {"name": name, "ok": False, "steps": done, "tail": tail, "sec": time.time() - t0}
    return {"name": name, "ok": True, "steps": done, "sec": time.time() - t0}


@app.function(image=image, volumes=VOLS, timeout=3600)
def run_cpu(cmd: list[str]) -> str:
    c = [sys.executable, *cmd[1:]] if cmd[0] == "python" else cmd
    r = subprocess.run(c, capture_output=True, text=True)
    runs.commit()
    return r.stdout[-6000:] + r.stderr[-3000:]


@app.local_entrypoint()
def sweep(tier: str = "clip", seeds: str = "0 1", steps: int = 30_000, n_points: int = 8192, samples: int = 5,
          sample_steps: int = 100, participants: str = "sub01", encoder_run: str = "", models: str = "",
          prefix: str = "bench", gpu: str = DEFAULT_GPU, pvcnn_gpu: str = "A100-80GB", dry_run: bool = False,
          deployed: bool = False):
    """--deployed spawns on the deployed app (`modal deploy scripts/modal_n3d.py` first) and returns at once;
    those calls outlive this client, unlike `modal run --detach`, which keeps only the last spawned call alive."""
    from n3d import sweep as sw
    js = sw.jobs(tier, tuple(int(s) for s in seeds.split()), steps, n_points, samples, sample_steps,
                 participants=tuple(participants.split()), encoder_run=encoder_run or None,
                 models=tuple(models.split()) if models else tuple(sw.MODELS), prefix=prefix)
    for j in js:
        print(f"{j.name:34s} gpu={pvcnn_gpu if j.needs_cuda else gpu}  ({len(j.cmds)} commands)")
    if dry_run:
        return
    # PVCNN at batch 48 peaks at 26-29 GB (film) and 33-40 GB (concat, 1027-channel voxel grids): too big for 24 GB
    fn = modal.Function.from_name("n3d", "run_job") if deployed else run_job
    calls = [fn.with_options(gpu=pvcnn_gpu if j.needs_cuda else gpu).spawn(j.name, j.cmds) for j in js]
    if deployed:
        ids = {j.name: c.object_id for j, c in zip(js, calls)}
        out = ROOT / "runs" / f"calls_{prefix}_{tier}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(ids, indent=1))
        print(f"spawned {len(ids)} calls on deployed app; ids in {out}\n"
              f"check: modal run scripts/modal_n3d.py::status --calls {out}")
        return
    report(calls)


@app.local_entrypoint()
def status(calls: str):
    """Poll calls spawned with sweep --deployed (no waiting): running / ok / FAIL with log tail."""
    ids = json.loads(Path(calls).read_text())
    for name, cid in ids.items():
        try:
            res = modal.FunctionCall.from_id(cid).get(timeout=0)
        except TimeoutError:
            print(f"run  {name}"); continue
        except Exception as e:  # noqa: BLE001
            print(f"EXC  {name}  {repr(e)[:300]}"); continue
        print(f"ok   {name}  {res['sec'] / 60:.1f} min" if res["ok"] else f"FAIL {name}\n  " + "\n  ".join(res["tail"]))


def report(calls):
    for c in calls:
        try:
            res = c.get()
        except Exception as e:  # noqa: BLE001
            print("EXCEPTION", repr(e)[:500]); continue
        if res["ok"]:
            print(f"ok   {res['name']}  {res['sec'] / 60:.1f} min")
        else:
            print(f"FAIL {res['name']}\n  " + "\n  ".join(res["tail"]))


@app.local_entrypoint()
def smoke(gpu: str = "A100-80GB", steps: int = 200):
    """Compile PVCNN ops (cached afterwards) and time both PVCNN modes at 2048 and 8192 points."""
    js = []
    for mode in ("concat", "film"):
        for n in (2048, 8192):
            run = f"smoke_pv_{mode}_{n}"
            js.append((run, [["python", "-m", "n3d.diffusion", "train", "--run", run, "--restart", "--cond-source", "clip",
                              "--denoiser", "pvcnn", "--cond-mode", mode, "--n-points", str(n), "--max-steps", str(steps),
                              "--log-every", "50", "--eval-every", str(steps), "--n-eval", "8", "--eval-steps", "20"]]))
    fn = run_job.with_options(gpu=gpu)
    first = fn.spawn(*js[0])  # builds the CUDA extension once before the rest start in parallel
    report([first])
    report([fn.spawn(*j) for j in js[1:]])
    print(run_cpu.remote(["bash", "-c", "for f in /runs/smoke_pv_*/metrics.jsonl; do echo $f; tail -1 $f; done"]))


@app.local_entrypoint()
def cmd(args: str, gpu: str = DEFAULT_GPU):
    """Run one arbitrary command on a GPU, e.g. --args "python -m n3d.baselines --run baselines_sub01"."""
    print(run_job.with_options(gpu=gpu).remote("cmd_" + str(abs(hash(args)) % 10**8), [shlex.split(args)]))


@app.local_entrypoint()
def aggregate(prefix: str = "bench", baselines: str = "", tag: str = ""):
    extra = (["--baselines", baselines] if baselines else []) + (["--tag", tag] if tag else [])
    print(run_cpu.remote(["python", "-m", "n3d.aggregate", "--prefix", prefix, *extra]))


@app.local_entrypoint()
def audit(participants: str = "sub01 sub02 sub03 sub04 sub05", seeds: str = "0 1", epochs: int = 200,
          prefix: str = "audit", gpu: str = DEFAULT_GPU, summarize: bool = False):
    """Encoder replication audit (n3d.audit): one deployed GPU job per participant. --summarize prints the table."""
    if summarize:
        print(run_cpu.remote(["python", "-m", "n3d.audit", "--summarize", "--prefix", prefix]))
        return
    from n3d import audit as au
    js = au.jobs(tuple(participants.split()), tuple(int(s) for s in seeds.split()), epochs, prefix)
    fn = modal.Function.from_name("n3d", "run_job").with_options(gpu=gpu)
    ids = {j.name: fn.spawn(j.name, j.cmds).object_id for j in js}
    out = ROOT / "runs" / f"calls_{prefix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(ids, indent=1))
    print(f"spawned {len(ids)} calls; check: modal run scripts/modal_n3d.py::status --calls {out}")
