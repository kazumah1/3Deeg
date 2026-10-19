# Running experiments on Colab

The code is in `n3d/`; notebooks only call it. Open
[`colab/3deeg_pipeline.ipynb`](3deeg_pipeline.ipynb) in Colab (File > Open notebook > GitHub,
`kazumah1/3deeg`, branch `colab-pipeline`) or paste its cells into your own notebook.

## Where things live

| What | Path | Notes |
| --- | --- | --- |
| Downloaded archives | `MyDrive/3deeg/data/neuro-3d/archives/` | Pinned HF revision, fetched once |
| Packed arrays | `MyDrive/3deeg/data/neuro-3d/packed/` | What every experiment loads |
| Runs | `MyDrive/3deeg/runs/<run>/` | `config.json`, `metrics.jsonl`, `best.pt`, plus the checkpoints below |
| Compiled PVCNN ops | `MyDrive/3deeg/torch_extensions/` | Reused while torch/CUDA versions match |
| Upstream Neuro-3D code | `/content/n3d_cache/` | Cloned at a pinned commit each runtime; not vendored (no licence) |

## Checkpoints and disconnects

Re-running a training cell resumes automatically; pass `--restart` to start over.

| File | Written | Contents | Kept |
| --- | --- | --- | --- |
| `last.pt` | encoder: every epoch; diffusion: every `--ckpt-every` steps (1,000) | model + optimizer + scheduler + RNG | newest only |
| `last_prev.pt` | each time `last.pt` is replaced | the previous `last.pt` | newest only |
| `snapshots/` | encoder: every `--snapshot-every` epochs (20); diffusion: every `--snapshot-every` steps (5,000) | full state, resumable | encoder all; diffusion newest `--keep-snapshots` (3) |
| `milestones/` | diffusion: every `--milestone-every` steps (10,000) | weights only, for `sample --ckpt milestones/stepXXXXXXX.pt` | all |

Every checkpoint is written to a temp file and renamed into place, so a kill mid-save
cannot corrupt it. Resume tries `last.pt`, then `last_prev.pt`, then snapshots newest first,
skipping any file that fails to load (for example one Drive never finished uploading).
Logs are opened and closed per line because Drive uploads a file only after it is closed.

Drive syncs in the background, so a runtime that dies abruptly can lose the last few
minutes of writes. That is why `last_prev.pt` and the snapshots exist: the worst case is
falling back one snapshot interval. After a long run, `drive.flush_and_unmount()` forces
the final upload before you disconnect.

## Commands

```sh
python -m n3d.download --participants sub01
python -m n3d.encoder   train    --run enc_sub01                 # upstream encoder recipe, val-selected
python -m n3d.encoder   diagnose --run enc_sub01                 # stream ablations, shuffle, noise, probes
python -m n3d.encoder   embed    --run enc_sub01                 # cache f_g / f_a
python -m n3d.baselines --run baselines_sub01 --encoder-run enc_sub01
python -m n3d.diffusion train  --run dm_sub01 --encoder-run enc_sub01        # PVCNN, upstream recipe
python -m n3d.diffusion sample --run dm_sub01 --eval-split test --samples 5 --steps 100
python -m n3d.diffusion swap   --run dm_sub01 --eval-split test
```

Every command accepts `--tiny` (4 categories) for overfit checks. Diffusion accepts
`--denoiser pointnet`, `--cond-mode film`, `--cond-source clip|category`, `--width`, `--voxel-res`.

## Conventions

* Splits by object variant: train 00-06, val 07, test 08-09 (`--split upstream` = train 00-07, no val).
  Upstream selected its encoder checkpoint on test accuracy; we select on val.
* Evaluation EEG averages each object's repeats, as upstream does. `diagnose` also reports single trials.
* Chamfer: symmetric mean squared, on upstream-normalised clouds (centred, unit max radius). F-score at
  0.01 / 0.02 / 0.05. Mean over samples and best-of-k are always reported separately.
* Geometry defaults to `point_cloud_simple` (what upstream trains on); `--geometry point_cloud` for the
  version used by `benchmark/`.

## Testing without a GPU

`python -m n3d.synthetic /tmp/fake` writes a fake packed dataset with the real shapes. Point
`N3D_DATA` at it and run any command with `--denoiser pointnet` to check the plumbing on CPU.
