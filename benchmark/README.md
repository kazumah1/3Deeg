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
| `scripts/benchmark_k_oriented_cuboids.py` | Fits and scores the hardcoded k=4 cuboid baseline. |
| `k_oriented_cuboid/k4_ten_object/` | Precomputed k=4 parameters, scores, and summary for the explorer. |
| `k_oriented_cuboid_fitting.md` | k=4 method and run guide. |

Run the prepared ten-object cuboid benchmark with:

```sh
.venv/bin/python benchmark/scripts/benchmark_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv
```

Run the k=4 comparison with:

```sh
.venv/bin/python benchmark/scripts/benchmark_k_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv
```
