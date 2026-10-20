# K-superquadric fitting

Each superquadric run begins from the matching precomputed k-oriented cuboid fit. For every component, the cuboid center becomes the initial translation, its PCA axes become the initial orientation, and its half extents become the three initial superquadric scales.

The superquadric surface is parameterized by scales `(a1, a2, a3)` and two shape exponents `(epsilon1, epsilon2)`. Both exponents start at 1.0, the rounded-cuboid setting. Smaller values make the surface progressively more pinched or box-like.

## Benchmark procedure

1. For each `k` from 1 through 8, form the matching cuboid decomposition.
2. Sweep the 25 shared exponent pairs where `epsilon1, epsilon2` are each in `{0.2, 0.4, 0.6, 0.8, 1.0}`. The pair is shared by every component, avoiding an exponential `25^k` search.
3. For each candidate, refine each component's translation, rotation, and scale with bounded L-BFGS-B optimization.
4. Allocate exactly 8,192 evaluation samples among components in proportion to estimated superquadric surface area. Select the shared pair using the symmetric squared Chamfer distance from the resulting union to the full object point cloud.
5. Starting from the winning shared pair, refine each component's two exponents continuously in `[0.1, 1.0]`, along with its pose and scale.
6. Report only the final union-to-full-cloud symmetric squared Chamfer distance. Component-local point sets are used during optimization for tractability, never as the reported benchmark metric.

Run one value with:

```sh
.venv/bin/python benchmark/scripts/benchmark_superquadrics.py --k 4
```

Run all displayed values with:

```sh
for k in {1..8}; do
  .venv/bin/python benchmark/scripts/benchmark_superquadrics.py --k "$k"
done
```

Each output directory, `benchmark/superquadric/k{value}_ten_object/`, contains `fitted_superquadrics.npz`, `per_object.csv`, and `summary.json`. The NPZ file supplies the visualizer with centers, axes, scales, and fitted `epsilon1`/`epsilon2` values.
