# Project Context: 3D Object Generation from EEG

## Mission

Build and evaluate systems that generate a 3D object directly from EEG recorded while a person views that object. The aim is to test whether we can design a suitable generator, not to route brain activity through a language model.

Pipeline:
1. A participant views an object while EEG is recorded.
2. An existing EEG encoder (from Neuro-3D/prior work) maps the signal to an embedding intended to represent the viewed object.
3. A 3D generator maps that embedding directly to geometry.

Treat the encoder as a mostly fixed input component initially. The main open research question is how to generate geometry from its embedding.

## Project Requirements and Design Space

- No text captioning or language model in the EEG-to-geometry path. Captioning can discard geometric detail that captions do not mention.
- Neuro-3D's ground-truth geometry is point-cloud data. Compare point-cloud generation with a compact primitive representation, if the primitive targets can be fitted accurately enough.
- Candidate generators include direct regression and diffusion. Neuro-3D applies diffusion to point clouds; this project should test whether a much smaller primitive-parameter representation makes diffusion simpler while retaining quality.
- Compare EEG conditioning methods such as concatenation, cross-attention, and FiLM. The best choice may depend on the generator.
- A usable 3D visualization/debugging tool is an early requirement. Inspect generated geometry directly, not only scalar metrics such as Chamfer distance.
- Important open questions: regression collapse to average shapes; fidelity loss from primitive fitting; which conditioning mechanism works best; and whether failures come from the encoder, conditioning, or generator family.

## Dataset: Immediate Starting Task

Set up the public Neuro-3D / EEG-3D dataset from Hugging Face:

- Dataset: https://huggingface.co/datasets/guozq21/neuro-3D
- Upstream code and data-preparation notes: https://github.com/gzq17/neuro-3D
- Paper: https://arxiv.org/abs/2411.12248

Confirmed repository facts from the Hugging Face file manifest:

- The repository is about 14.5 GB and consists of files/archives, not a directly loadable Hugging Face `datasets` table. The Hub dataset viewer does not recognize its contents as supported tabular data.
- EEG participant files are `EEGdata/sub01.zip` through `EEGdata/sub12.zip`, each about 712 MB.
- Core geometry archive: `point_cloud.zip`, about 102 MB.
- Other optional assets include `point_cloud_simple.zip` (~88 MB), `model3d.zip` (~2.4 GB), `image.zip` (~579 MB), `video_new.zip` (~2.75 GB), `clip_feature.pth` (~8.7 MB), `clip_feature_gray.pth`, labels, and descriptions.
- Upstream code README describes extracted EEG files such as `EEGdata/sub01/process_data_1s_250Hz.npy` and `process_data_6s_100Hz.npy`, alongside `point_cloud`, `video_new`, and `clip_feature.pth`. Its instructions appear older than the current Hub archive layout; inspect archive contents before assuming paths or preprocessing compatibility.
- A good initial download should be selective: one participant's EEG plus point clouds, roughly 814 MB before extraction. Do not pull all 14.5 GB by default.
- `clip_feature.pth` is optional for this project's direct EEG-to-geometry path. Do not introduce a text/CLIP intermediate into that path; only use optional pretrained features if a clearly justified experiment needs them.

## Team Dataset Handoff

**Neuro-3D / EEG-3D:** [dataset access instructions + where a locally preprocessed version will live](DATASETS.md).

- Selective setup is implemented in `scripts/setup_neuro3d.py`; see `README.md` for commands.
- `sub01` and core point clouds have been downloaded, extracted, and integrity-verified.
- Available upstream processed EEG: `data/neuro-3d/extracted/sub01/`.
- Geometry targets: `data/neuro-3d/extracted/point_cloud/`.
- Trial-to-object index: `data/neuro-3d/processed/v1/sub01_trial_to_object_index.csv` (generated locally from the verified labels; 1,728 rows).
- Future signal preprocessing: `data/neuro-3d/processed/v1/` (reserved; not generated).
- Default local root: `<checkout>/data/neuro-3d/` (download separately on each machine).
- No shared team storage has been configured; teammates can reproduce the subset locally.
- EEG archives extract to `sub01/`, rather than the older upstream `EEGdata/sub01/` layout.
- Setup preserves upstream files and does not add normalization, dataset splits, or an encoder/training stack. `scripts/build_trial_index.py` derives a local mapping from every trial to its source object and upstream-compatible split role.
- Dataset files and caches are ignored by Git.
- Local explorer: `.venv/bin/python scripts/explore_neuro3d.py`, then open `http://127.0.0.1:8765`; see `EXPLORING.md`.
- NumPy is installed in `.venv`; exploration requirements are in `requirements-explore.txt`.
- Label-to-geometry mapping strips the leading category ID (e.g. `01_airplane_06` → `airplane_06.npy`); all 720 objects map successfully for sub01.
- Both EEG windows were checked for finite values. No model or additional preprocessing has been run.

## Next experiment: primitive fitting

- A fixed ten-object, geometry-only benchmark subset is prepared by `scripts/prepare_primitive_benchmark.py`. It uses held-out `08` variants from ten distinct categories and writes a manifest plus packed XYZ/RGB arrays under `data/neuro-3d/processed/v1/`.
- Compare one oriented cuboid, `k` oriented cuboids, a superquadric, Gaussian-mixture ellipsoids, and medial spheres.
- Primary score: symmetric squared Chamfer distance between equal-count primitive surface samples and original XYZ points. Also report visual comparison, fitting runtime, convergence status, and loss trace.
- Do not compare methods until component budgets, initialization, seeds, stopping criteria, and surface-sampling count are recorded. This is a geometry-approximation benchmark, not an EEG-decoding experiment.

## Saved Checkpoint — October 7, 2026

- User reviewed the local EEG/3D explorer and approved its current appearance; work stopped here at their request.
- Explorer code and team walkthrough are saved in `scripts/explore_neuro3d.py`, `explorer/index.html`, and `EXPLORING.md`.
- Launch: `.venv/bin/python scripts/explore_neuro3d.py`; open `http://127.0.0.1:8765`. Restart the server if needed in a later session.
- Browser rendering was inspected; geometry, EEG trace, and heatmap loaded successfully. API checks covered both recording windows, label mapping, point counts, and invalid requests.
- Available data remains sub01 plus core point clouds. The local trial-to-object index has been generated; no encoder, generator, or signal preprocessing outputs have been created.
- Suggested next step when resuming: establish encoder inputs and checkpoint availability, then inspect embeddings before selecting a geometry generator.

## Saved Discussion — October 8, 2026

Read [Neuro-3D, arXiv v3](https://arxiv.org/pdf/2411.12248v3), including supplementary material.

- Dataset: 72 categories, 10 objects each (720 instances), 12 participants. Per participant, 576 training objects have two viewing repetitions and 144 held-out test objects have four: 1,728 trials. Categories remain shared across training and testing.
- Each trial has paired static-image and rotating-video EEG windows. Local arrays contain 64 channels, with 250 samples at 250 Hz and 600 samples at 100 Hz, respectively.
- Method: EEG fusion produces 1,024-dimensional geometry/appearance features; diffusion generates 8,192 points, followed by dominant-color prediction. CLIP image features supply training supervision; no caption/LLM inference step is described.
- Reported accuracy: 72-category EEG classification 5.91% versus 1.39% chance. Reconstruction 2-way accuracy averages 55.81%; best-of-five is 72.08%. These measure different tasks. Testing averages four EEG repetitions.
- User's concern: recognizable outputs may depend heavily on learned category shapes. Held-out instances limit exact training-object memorization but do not establish how much instance geometry comes from EEG. This is an interpretation to test, not proven memorization.
- Proposed controls: swaps between objects within a category, shuffled EEG, and category-only/typical-shape baselines. No experiments have run.
- Before encoder reuse: audit code, weights, inputs, and the paper's 250 Hz preprocessing versus local dynamic 100 Hz arrays.

Only data setup and visualization are implemented locally. Upstream decoding code, checkpoints, and the training stack have not been installed. The existing explorer and dataset files remain saved.

## Citation

If publishing work based on the dataset, cite Neuro-3D: Guo et al., “Neuro-3D: Towards 3D Visual Decoding from EEG Signals,” CVPR 2025; arXiv:2411.12248.
