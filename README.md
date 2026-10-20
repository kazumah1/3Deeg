# 3Deeg: EEG and 3D object explorer
Inspect Neuro-3D / EEG-3D recordings alongside their corresponding 3D objects. The dashboard lets you rotate colored point clouds, view EEG channels and heatmaps, and compare repeated trials and recording windows.

## Quick start

The explorer and trial index require Python 3.12+ and curl. From the cloned repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-explore.txt
.venv/bin/python scripts/setup_neuro3d.py download
.venv/bin/python scripts/setup_neuro3d.py extract
.venv/bin/python scripts/setup_neuro3d.py verify
.venv/bin/python scripts/build_trial_index.py
.venv/bin/python scripts/explore_neuro3d.py
```

Open **http://127.0.0.1:8765**. The initial setup downloads sub01 EEG and all core point clouds (about 814 MB); allow at least 2 GB free. Data is downloaded separately and excluded from Git. No EEG decoding model is running.

The explorer can overlay precomputed PCA-oriented cuboids for each k from 1 through 8 on the prepared ten-object benchmark. Generate or refresh those fits with `for k in {1..8}; do .venv/bin/python benchmark/scripts/benchmark_k_oriented_cuboids.py --manifest benchmark/primitive_10_objects.csv --k "$k"; done`.

It also exposes superquadrics for k=1 through k=8. Each starts from the corresponding cuboid decomposition, evaluates the 25 shared ε₁/ε₂ pairs, then refines ε values per component. Generate them with `for k in {1..8}; do .venv/bin/python benchmark/scripts/benchmark_superquadrics.py --k "$k"; done`.

**Understand the baseline:** [developer guide to oriented cuboid fitting](benchmark/oriented_cuboid_fitting.md). All benchmark inputs, methods, results, and previews live in [benchmark/](benchmark/).

## Dataset setup

**Start exploring:** [interactive EEG and point-cloud walkthrough](EXPLORING.md).

**Build the trial-to-object table:** `.venv/bin/python scripts/build_trial_index.py` writes `data/neuro-3d/processed/v1/sub01_trial_to_object_index.csv`, mapping every zero-based EEG trial to its object label, repeat, geometry file, and upstream-compatible train/test role.

**Team handoff:** [Neuro-3D / EEG-3D access instructions and local preprocessing locations](DATASETS.md).

Download a selective subset of [Neuro-3D / EEG-3D](https://huggingface.co/datasets/guozq21/neuro-3D), independently of its training stack. Requires Python **3.11+**, `curl`, network access, and disk space. No pip packages, PyTorch, CUDA, captions, or CLIP features are needed. Commands below run from the workspace root.

```sh
# Default: sub01 plus core point clouds. Downloads only; rerun to resume.
python3 scripts/setup_neuro3d.py download
python3 scripts/setup_neuro3d.py inspect
# Extraction is a separate, explicit action.
python3 scripts/setup_neuro3d.py extract
python3 scripts/setup_neuro3d.py verify

# Select participants and a different destination (repeat options for each action).
python3 scripts/setup_neuro3d.py download --participants sub02 sub03 --data-dir /path/to/neuro-3d
python3 scripts/setup_neuro3d.py extract --participants sub02 sub03 --data-dir /path/to/neuro-3d
# Explicitly opt into all 12 participants, plus point clouds:
python3 scripts/setup_neuro3d.py download --all-participants
```

The default revision is pinned to `31ebd72706a6b6b40bfe77022f9ca48a7a12dfdd`, inspected on October 7, 2026. `download` fetches the Hub file manifest and saves it as `manifest.json`, then downloads only selected archives. Use `--revision COMMIT` with a separate destination to change revisions. Later offline actions use the saved manifest. Completed archives are checked against Hub sizes and SHA-256 hashes. Interrupted transfers remain in `.zip.part` files; curl resumes them on the next run. Failed transfers or checksum mismatches stop the command. A checksum mismatch requires removing the reported archive/partial file and downloading again.

Default layout, preserving archive paths:

```text
data/neuro-3d/
  manifest.json
  archives/
    point_cloud.zip
    EEGdata/sub01.zip
  extracted/
    point_cloud/airplane_00.npy   # and other geometry files
    sub01/
      name_label.npy
      process_data_1s_250Hz.npy
      process_data_6s_100Hz.npy
```

**Compatibility detail:** the EEG ZIP contains `sub01/`, not `EEGdata/sub01/`. Extraction preserves that layout; downstream code must use `extracted/sub01` rather than assume the older [upstream README](https://github.com/gzq17/neuro-3D) paths. No renaming or preprocessing occurs. Extraction may overwrite existing files from selected archives. ZIP paths and destination symlinks are checked before extraction.

The default download is **813,706,399 bytes (~814 MB)**. Extraction adds **925,715,184 bytes (~926 MB)**; allow at least **2 GB free** for both. All participant archives plus geometry need approximately 8.65 GB compressed and another 9.2 GB extracted; allow at least 20 GB free. Other optional assets are never fetched. The whole Hub repository is approximately 14.5 GB, but this command does not download it wholesale.

`inspect` lists member counts, uncompressed sizes, and sample paths without extraction. `verify` checks archive SHA-256 hashes and, if `extracted/` exists, requires every selected member to exist with matching size and CRC. If extraction was interrupted, rerun `extract` before `verify`. Local inspect/extract/verify do not need network access.

## Validation performed

Downloaded, inspected, extracted, and verified sub01 and point clouds at the pinned revision. Geometry has 865 ZIP members (including its directory); EEG has four. NPY headers inspected without loading arrays:

| File | Shape | dtype |
| --- | --- | --- |
| `sub01/process_data_1s_250Hz.npy` | `(1728, 64, 250)` | float64 |
| `sub01/process_data_6s_100Hz.npy` | `(1728, 64, 600)` | float64 |
| `sub01/name_label.npy` | `(1728,)` | Unicode, 23 characters |
| `point_cloud/airplane_00.npy` | `(8192, 6)` | float32 |

This validates packaging and file integrity, not EEG preprocessing semantics, label-to-geometry alignment, or compatibility with a particular encoder. Other participants have not been downloaded or inspected.

Workspace data, ZIPs, partial transfers, `.cache/`, and virtual environments are ignored by `.gitignore`. Custom destinations inside the workspace must also be ignored before committing. No dataset files should be committed.

If publishing work using this dataset, cite Guo et al., “Neuro-3D: Towards 3D Visual Decoding from EEG Signals,” CVPR 2025; [arXiv:2411.12248](https://arxiv.org/abs/2411.12248).
