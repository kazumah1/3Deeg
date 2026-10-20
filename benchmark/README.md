# Benchmark assets

This directory is the single home for primitive-fitting benchmark inputs, methods, results, and visual previews. It contains derived benchmark artifacts only; the downloaded Neuro-3D source data remains under `data/neuro-3d/`.

| Path | Contents |
| --- | --- |
| `primitive_10_objects.csv` | Fixed held-out ten-object manifest. |
| `primitive_10_objects.npz` | The corresponding XYZ and RGB point clouds. |
| `scripts/prepare_primitive_benchmark.py` | Rebuilds the fixed subset. |
| `scripts/benchmark_oriented_cuboids.py` | Fits and scores oriented-cuboid baselines. |
| `oriented_cuboid/ten_object/` | Ten-object cuboid parameters, scores, and run summary used by the explorer. |
| `oriented_cuboid/training/` | Full training-set cuboid results and preview. |
| `oriented_cuboid_fitting.md` | Conceptual and implementation guide. |
| `scripts/benchmark_k_oriented_cuboids.py` | Fits and scores deterministic k-oriented cuboid baselines. |
| `k_oriented_cuboid/k{1..8}_ten_object/` | Precomputed k=1 through k=8 parameters, scores, and summaries for the explorer. |
| `k_oriented_cuboid_fitting.md` | k-oriented cuboid method and run guide. |
| `scripts/benchmark_superquadrics.py` | Two-stage k-superquadric benchmark initialized from the matching cuboid fit. |
| `superquadric/k{1..8}_ten_object/` | Precomputed k=1 through k=8 superquadric parameters, scores, and summaries for the explorer. |
| `superquadric_fitting.md` | Superquadric parameterization, objective, and benchmark procedure. |

Run the prepared ten-object cuboid benchmark with:

```sh
.venv/bin/python benchmark/scripts/benchmark_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv
```

Run a k-oriented cuboid comparison with:

```sh
.venv/bin/python benchmark/scripts/benchmark_k_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv --k 4
```

Run the matching superquadric benchmark with:

```sh
.venv/bin/python benchmark/scripts/benchmark_superquadrics.py --k 4
```
