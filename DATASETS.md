# Team dataset access

**Neuro-3D / EEG-3D:** [dataset access instructions and local preprocessing locations](#neuro-3d--eeg-3d).

## Neuro-3D / EEG-3D

To inspect EEG recordings alongside 3D objects, follow the [first-session walkthrough](EXPLORING.md).

Source: [Hugging Face dataset](https://huggingface.co/datasets/guozq21/neuro-3D). Preparation reference: [upstream code](https://github.com/gzq17/neuro-3D). Our setup uses pinned revision `31ebd72706a6b6b40bfe77022f9ca48a7a12dfdd`.

### Access on your machine

From a copy of this workspace, with Python 3.11+ and curl installed:

```sh
python3 scripts/setup_neuro3d.py download
python3 scripts/setup_neuro3d.py extract
python3 scripts/setup_neuro3d.py verify
```

This fetches **sub01 and core point clouds only**, resumes interrupted downloads, and checks file integrity. Allow at least **2 GB free**. No training dependencies or Hugging Face account were needed for the validated download.

For other participants, repeat `--participants sub02 sub03` on each command. For all 12, repeat `--all-participants` on each command and allow at least 20 GB free. Use `--data-dir /your/data/location/neuro-3d` on each command to change the destination. Full options and validation details are in [README.md](README.md).

### Where the data lives

Paths below are relative to the workspace root, or to the directory selected by `--data-dir`:

| Data | Default location | Status |
| --- | --- | --- |
| Original downloaded archives and source manifest | `data/neuro-3d/archives/`, `data/neuro-3d/manifest.json` | Downloaded for sub01 and point clouds |
| Upstream processed EEG arrays and trial labels | `data/neuro-3d/extracted/sub01/` | Present and integrity-verified |
| Geometry targets | `data/neuro-3d/extracted/point_cloud/` | Present and integrity-verified |
| Trial-to-object index | `data/neuro-3d/processed/v1/sub01_trial_to_object_index.csv` | Generated locally from verified labels |
| Future project-specific preprocessing | `data/neuro-3d/processed/v1/` | Reserved; no signal transformation generated yet |

After extraction, upstream processed EEG is available at `<checkout>/data/neuro-3d/extracted/sub01/` (or `<data-dir>/extracted/sub01/` for a custom destination).

It contains `process_data_1s_250Hz.npy`, `process_data_6s_100Hz.npy`, and `name_label.npy`. Geometry is alongside it in `extracted/point_cloud/`. These are the arrays distributed by the dataset authors. Our setup extracts them unchanged; it has not run additional normalization, train/test splitting, encoder embedding extraction, or EEG-to-geometry alignment validation.

Project-specific outputs will live at **`data/neuro-3d/processed/v1/`**, corresponding to **`<checkout>/data/neuro-3d/processed/v1/`**. When preprocessing is implemented, each version should include its source revision, participant list, transformation settings, split assignments, and reproduction command. Preserve the extracted source files and use a new version directory when the preprocessing changes.

Build the first derived artifact with `.venv/bin/python scripts/build_trial_index.py`. It creates a row for every EEG trial, with its source label, category, variant, repeat ordinal, matching geometry path, and the upstream-compatible `train`/`test` role. The `08` and `09` variants are marked `test`; all other variants are marked `train`. See [EXPLORING.md](EXPLORING.md#trial-to-object-index) for the command and field semantics.

### Team availability

Each checkout uses its own local data directory; no shared server or team storage location has been configured. Teammates can reproduce the validated subset using the commands above. The status entries above describe the initial development validation, not data included in a fresh checkout. Dataset files are excluded from Git, so a repository checkout alone does not contain them. A future shared copy should publish its storage address and access instructions here.

For published work, cite Guo et al., “Neuro-3D: Towards 3D Visual Decoding from EEG Signals,” CVPR 2025; [arXiv:2411.12248](https://arxiv.org/abs/2411.12248).
