# Primitive-fitting benchmark

This fixed ten-object subset establishes the first comparison of compact geometry representations. It contains only held-out `08` variants, across ten categories and shape families, so fitting quality is not evaluated on a single type of shape.

## Prepare the subset

```sh
.venv/bin/python scripts/prepare_primitive_benchmark.py
```

The command writes two local, rebuildable files under `data/neuro-3d/processed/v1/`:

- `primitive_benchmark_10_objects.csv` records object identity, shape family, four associated sub01 trial indices, and source point-cloud path.
- `primitive_benchmark_10_objects.npz` packs raw XYZ `(10, 8192, 3)` and RGB `(10, 8192, 3)` arrays for fitting and visualization. Fitting uses XYZ only; RGB remains available for side-by-side displays.

The selected objects are airplane, banana, bottle, chair, Christmas tree, elephant, guitar, motorcycle, ring, and spider. The manifest is the authoritative order; do not replace a selected object without versioning the benchmark.

## Methods to compare

Evaluate each object with the following representations:

1. One oriented cuboid over the complete object.
2. `k` oriented cuboids.
3. One superquadric.
4. Gaussian-mixture ellipsoids.
5. Medial spheres.

Choose and document the `k` values, component budget, initialization, optimizer, stopping rule, random seeds, and surface-sampling count before running the benchmark. Give methods comparable parameter/component budgets where their parameterizations differ.

## Primary metric and reporting

Use symmetric squared Chamfer distance on XYZ surface samples as the primary fitting loss and score. For primitive surface samples `P` and original cloud points `Q`:

`CD(P, Q) = mean(min(||p - q||²)) + mean(min(||q - p||²))`.

Use the same fixed sample count and seeds for every method/object evaluation. Report per-object and mean Chamfer distance, wall-clock fitting time, convergence status and loss trace, plus matched-view visualizations of the primitive samples and original colored point cloud. Visual quality is a qualitative companion to the metric; it should call out thin structures, concavities, disconnected components, and topology failures.

This benchmark evaluates geometric approximation only. It does not fit EEG, train a generator, or establish decoder performance.
