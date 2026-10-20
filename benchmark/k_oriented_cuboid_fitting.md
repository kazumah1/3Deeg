# Four oriented cuboids

This benchmark extends the single-cuboid baseline with four oriented cuboids (`k=4`). It first partitions an object's XYZ points with deterministic Lloyd k-means, initialized by farthest-point selection. Each cluster then receives its own PCA-oriented enclosing cuboid.

The four cuboid surfaces are sampled in proportion to their total surface areas, concatenated, and scored against the original point cloud using symmetric squared Chamfer distance. The result records the k-means iteration count and convergence status in addition to fitting and evaluation time.

Run the fixed ten-object subset with:

```sh
.venv/bin/python benchmark/scripts/benchmark_k_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv
```

Results are stored in `benchmark/k_oriented_cuboid/k4_ten_object/`. The current implementation intentionally hardcodes `k=4`. The explorer exposes it as **k=4 oriented cuboids** beside the single-cuboid baseline; additional values will be added only after their artifacts are precomputed and registered.
