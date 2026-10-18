"""Neuro-3D reproduction, diagnostics and experiment tooling for Colab.

Modules
-------
config     paths, pinned revisions, split rules
download   selective Hugging Face download (Colab only; needs huggingface.co)
data       arrays, splits and torch datasets built from the extracted files
upstream   clones the pinned upstream Neuro-3D repo and imports its models
metrics    Chamfer distance and F-score
encoder    train / evaluate / probe the upstream EEG encoder
denoisers  PVCNN (upstream, CUDA) and PointNet (pure torch) noise predictors
diffusion  train / sample the conditional point-cloud diffusion model
baselines  geometry-only reference Chamfer scores and classify-then-template
"""
