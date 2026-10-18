# Running experiments on Colab

The code is in `n3d/`; notebooks only call it. Open
[`colab/3deeg_pipeline.ipynb`](3deeg_pipeline.ipynb) in Colab (File > Open notebook > GitHub,
`kazumah1/3deeg`, branch `colab-pipeline`) or paste its cells into your own notebook.

## Where things live

| What | Path | Notes |
| --- | --- | --- |
| Downloaded archives | `MyDrive/3deeg/data/neuro-3d/archives/` | Pinned HF revision, fetched once |
| Packed arrays | `MyDrive/3deeg/data/neuro-3d/packed/` | What every experiment loads |
| Runs | `MyDrive/3deeg/runs/<run>/` | `config.json`, `metrics.jsonl`, `last.pt` (resume), `best.pt` |
| Compiled PVCNN ops | `MyDrive/3deeg/torch_extensions/` | Reused while torch/CUDA versions match |
| Upstream Neuro-3D code | `/content/n3d_cache/` | Cloned at a pinned commit each runtime; not vendored (no licence) |

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
