# Packed data migration — 2026-11-18

Destination: private Hugging Face dataset [`kazumah/3deeg-packed`](https://huggingface.co/datasets/kazumah/3deeg-packed).

Pinned data revision: `0107f64d1d6077f9b36b8e6fc688bf285e8246d4`.

The repository contains 17 data files, 4,800,155,977 bytes: 12 subject EEG NPZs, two geometry arrays, object metadata, video embeddings and color labels. A SHA-256 manifest accompanies the files. Sixteen files were taken from the existing local packed cache; the missing full geometry array was downloaded from the connected Drive packed folder. No original raw archives, licensed ShapeNet meshes, fMRI data, trained models, experiment results, or credentials were uploaded.

## Verification requirements

- Confirm the dataset repository is private and anonymous download is denied.
- Download all files into a fresh local directory and verify SHA-256 hashes.
- Validate all 12 subject arrays and shared arrays; load actual CPU dataset items through the existing training data loader and check object-disjoint 504/72/144 train/validation/test splits.
- Run the same pinned downloader and CPU loader in a live Colab runtime using a read-only `HF_TOKEN` secret. A local notebook syntax check alone does not satisfy this requirement.

## Completed local checks

All 17 files were downloaded afresh from the pinned private repository and passed SHA-256 checks. All 12 subjects passed array validation and the actual CPU dataset loader, preserving 504/72/144 disjoint train/validation/test object splits. Both geometry choices passed. Anonymous access returned HTTP 401. Four integrity/selection tests and notebook schema/code compilation passed. No training or GPU work ran.

Local evidence is saved separately at `/Users/kazuma/3deeg-runs/hf-packed-migration-20261118/verification.json`. Check its current status before cleanup. Live Colab initially connected successfully but lacked the `HF_TOKEN` secret. Deletion readiness remains pending until both runtime checks pass.

## Storage behavior

Local: set `N3D_DATA` to the downloader's destination. Colab: data goes to `/content/3deeg-data` and is re-fetched after runtime resets. Downloads select requested subjects only. `N3D_RUNS` still points at the existing checkpoint directory; compiled extension caches remain on Drive. The migration does not move or remove checkpoints.

The Drive connector cannot download the 376 MB subject NPZs because it has a 256 MiB per-file limit. The local source and uploaded copies are hash-verified; that does not independently establish byte-for-byte equality to the Drive NPZ copies. Drive copies have matching filenames and sizes. No Drive files have been deleted.

## Cleanup boundary

Do not delete anything while verification is pending. After both checks pass, only the duplicated `data/neuro-3d/packed/` data and recoverable upstream subject ZIPs are cleanup candidates. Keep `runs/`, encoder embeddings and checkpoints, locked splits/protocols, research reports, and any unique source or processed data. Existing notebooks must use the updated HF setup before their Drive data dependency is removed. The archive-based explorer remains a separate workflow.
