# First session: EEG and geometry

Start with one participant and view the ground-truth object alongside its EEG. This establishes what the generator will eventually receive and what it must reconstruct. The explorer and trial index require Python 3.12+.

## Start the explorer

From the workspace root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-explore.txt
.venv/bin/python scripts/explore_neuro3d.py
```

Open **http://127.0.0.1:8765**. Stop with Ctrl+C in the terminal running it. Use `--port 8766` if the port is occupied; `--participant sub02 --data-dir /path/to/neuro-3d` selects another downloaded participant. If data is missing, follow [DATASETS.md](DATASETS.md) first.

The explorer reads source arrays without modifying them. EEG is memory-mapped to avoid loading every trial at startup. NumPy is its only Python dependency; browser rendering uses local JavaScript and canvas, with no CDN or external services. It binds only to localhost.

## Try these steps

1. Start with **airplane**, variant **00**. Drag the point cloud to orbit on pitch and yaw, hold Shift while dragging to roll, scroll to zoom, and toggle point colors. It contains 8,192 XYZ/RGB points. This is the geometry target for the EEG trials, not a generated prediction.
2. Look at the solid EEG trace, then change **Channel index**. The heatmap shows all 64 channels; each row is centered and scaled by its own standard deviation for display. The trace retains stored amplitudes. Channel indices run from 0 to 63; electrode names and physical units are not yet verified.
3. Switch **Recording** between repeats. Solid green is the selected trial; dashed orange is the next repeat of that same object. The target object stays fixed while the brain recordings differ. Ask: how much variation must an encoder tolerate?
4. Switch between **1 second / 250 Hz** and **6 seconds / 100 Hz**. These are separate stored arrays; do not assume one is a crop of the other. Time axes begin at the first stored sample, not a verified stimulus onset.
5. Try another airplane variant, then a very different category such as **chair**. Compare the geometric changes. We will eventually evaluate whether generated points preserve these changes rather than converging to an average shape.

A recognizable target and a plausible EEG trace do not establish that an EEG decoder works. This tool is for inspecting inputs, target geometry, and repeat variability.

## Fitted cuboid overlay

The explorer recognizes the prepared ten-object primitive benchmark. Use the **Primitive benchmark** selector or **Next benchmark object** to move through the held-out `08` objects without manually changing category and variant. Selecting one automatically enables the amber **Fitted cuboid surface** overlay and shows that object's squared Chamfer score beside the trial information. The control is disabled for all other objects, which have no subset cuboid fit. Rebuild the fit with:

```sh
.venv/bin/python benchmark/scripts/benchmark_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv
```

## What we have checked

For sub01, both EEG arrays contain 1,728 finite trials with 64 channels. Labels identify 720 objects in 72 categories. Removing the leading category number maps every label to a geometry file: `01_airplane_06` → `airplane_06.npy`.

Objects ending in `00`–`07` have two repeats; those ending in `08`–`09` have four. Upstream holds out variants `08` and `09` for testing in [EEG_organization.py](https://github.com/gzq17/neuro-3D/blob/main/eeg_data_process/EEG_organization.py), and removes the category prefix when loading geometry in [EEGdataset.py](https://github.com/gzq17/neuro-3D/blob/main/eeg_data_process/EEGdataset.py). Our explorer uses core `point_cloud/`; that upstream loader defaults to `point_cloud_simple/`, which has not been downloaded.

Keep repeated trials of the same object together when designing evaluation splits. A random trial split would allow the same target object to appear on both sides. The explorer does not create train/test splits or project preprocessing outputs.

## Trial-to-object index

Build the verified, row-per-trial CSV index before creating any split or preprocessing output:

```sh
.venv/bin/python scripts/build_trial_index.py
```

This writes `data/neuro-3d/processed/v1/sub01_trial_to_object_index.csv`. Its 1,728 rows map the zero-based EEG trial index to the source label, parsed category and variant, per-object repeat ordinal, matching point-cloud file, and split role. `08` and `09` are marked `test` to mirror the upstream variant holdout; all other variants are marked `train`. The index is local, derived from the immutable extracted labels, and can be rebuilt after a fresh dataset setup. Pass `--participant sub02` or `--data-dir /path/to/neuro-3d` when applicable.

## Next experiment

After inspecting a few objects, use the trial-to-object index to establish encoder inputs and checkpoint availability. Inspect encoder embeddings before choosing the first geometry generator. Training an EEG-to-geometry model is a separate step; this session runs no encoder or generator.
