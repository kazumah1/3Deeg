# Oriented cuboid fitting: developer guide

This guide explains the first primitive-fitting baseline in `benchmark/scripts/benchmark_oriented_cuboids.py`. It fits one rectangular cuboid to an object point cloud, then measures how well the cuboid surface approximates the original geometry. It is a geometry-only baseline: EEG is not used in the fit.

## What the fit receives and produces

Each Neuro-3D object is an array with shape `(8192, 6)`:

- Columns `0:3` are XYZ coordinates.
- Columns `3:6` are RGB values.

The cuboid fit uses XYZ only. Its output is three pieces of data:

- `center`: the cuboid's world-space center, `(3,)`.
- `axes`: a `3 × 3` orthonormal rotation matrix.
- `extents`: length, width, and height along the rotated axes, `(3,)`.

Together, those parameters describe an oriented bounding box. They are saved in `fitted_cuboids.npz`; `per_object.csv` records the fitting and scoring metrics.

## Why an oriented cuboid

An axis-aligned box can only face the global X, Y, and Z directions. A car, guitar, or airplane can be diagonal in that coordinate system, which makes its axis-aligned box unnecessarily large. An oriented cuboid first estimates the object's principal directions, then makes the box face those directions.

This is intentionally a simple baseline. One cuboid has no way to model a hole, a curve, separated components, limbs, or thin appendages. Its value is that it is fast, deterministic, and exposes how much geometric detail a more flexible primitive method must recover.

## Step 1: center the cloud

For point cloud coordinates `xᵢ`, calculate the centroid:

`μ = (1 / N) Σ xᵢ`

Then work with centered coordinates:

`yᵢ = xᵢ − μ`

Centering separates translation from orientation. The centroid is only the starting reference point; the final cuboid center can move relative to it after the bounding ranges are found.

## Step 2: find the PCA axes

Compute the `3 × 3` covariance matrix of the centered coordinates:

`C = cov(y)`

Its eigenvectors are orthogonal directions of greatest, middle, and least point-cloud variance. The script sorts them from largest to smallest eigenvalue and stores them as the columns of `A`:

`A = [a₁ a₂ a₃]`

Because `A` is orthonormal, it is a rotation matrix: `AᵀA = I`. PCA gives the cuboid a data-driven orientation without any iterative optimization.

Important detail: an eigenvector can be sign-flipped without changing the same physical cuboid. For example, `a₁` and `−a₁` describe the same axis. Do not treat a sign change in saved axes as a changed fit.

## Step 3: transform into the cuboid frame

Express every centered source point in the PCA frame:

`zᵢ = yᵢ A`

In this local frame the proposed cuboid is axis-aligned. For each local coordinate dimension, find its minimum and maximum:

`l = min(zᵢ)`

`u = max(zᵢ)`

The box extents are:

`e = u − l`

The local center is `(l + u) / 2`. Convert it back to world coordinates:

`c = μ + ((l + u) / 2) Aᵀ`

This box encloses every original point. It is a PCA-aligned enclosing box, not necessarily the smallest-volume oriented bounding box and not a box optimized directly for Chamfer distance.

## Step 4: sample the cuboid surface

Chamfer distance compares point sets, so the analytic cuboid must be converted into points. The benchmark samples 8,192 points from its six faces—matching the source cloud's point count.

The six faces are selected in proportion to their areas. This matters: sampling every face equally would overrepresent small faces. After choosing a face, the other two local coordinates are sampled uniformly within their extent and the face coordinate is fixed at `−eⱼ / 2` or `+eⱼ / 2`.

For local surface point `q`, transform back to world coordinates:

`p = c + q Aᵀ`

The benchmark uses seed `20261014` plus the object's ordered index, so repeated runs use the same surface samples and yield comparable scores.

## Step 5: calculate symmetric squared Chamfer distance

Let `P` be sampled cuboid-surface points and `Q` be the original cloud's XYZ points. The reported loss is:

`CD(P, Q) = meanₚ min_q ||p − q||² + mean_q minₚ ||q − p||²`

Both terms matter:

- `P → Q` penalizes cuboid surface that lies where the object has no surface.
- `Q → P` penalizes object geometry that the cuboid fails to cover.

The implementation uses SciPy's `cKDTree` for exact nearest-neighbor lookup rather than allocating an `8192 × 8192` distance matrix. This keeps each evaluation fast and avoids large temporary memory allocations.

`per_object.csv` also includes Chamfer normalized by the fitted cuboid's squared diagonal. Raw squared Chamfer retains the original coordinate units; the normalized value is more useful when comparing objects with different physical scales.

## What “convergence” means here

This baseline does not iterate. PCA, coordinate bounds, and the resulting cuboid are closed-form calculations. The result therefore records:

- `convergence_status = closed_form_pca`
- `iterations = 1`

This should not be compared directly with the iterative convergence behavior of multi-cuboid, superquadric, Gaussian-mixture, or medial-sphere fitting. For those methods, record initialization, stopping conditions, loss trace, number of iterations, and failure modes separately.

## Running the benchmark

Run the fixed ten-object subset:

```sh
.venv/bin/python benchmark/scripts/benchmark_oriented_cuboids.py \
  --manifest benchmark/primitive_10_objects.csv
```

Omit `--manifest` to fit all 576 `00`–`07` training objects. The standard output contains aggregate metrics. The output directory contains:

| File | Purpose |
| --- | --- |
| `benchmark/oriented_cuboid/ten_object/per_object.csv` | Per-object timing, Chamfer scores, and closed-form convergence metadata. |
| `benchmark/oriented_cuboid/ten_object/fitted_cuboids.npz` | Labels plus numeric centers, axes, and extents for rendering or later analysis. |
| `benchmark/oriented_cuboid/ten_object/summary.json` | Aggregate run configuration and summary metrics. |

## How the explorer uses the fit

`scripts/explore_neuro3d.py` loads the ten benchmark cuboids at startup and exposes a fitted cuboid only for a matching benchmark label. The browser receives center, axes, extents, and the object's Chamfer score. It normalizes the cuboid in exactly the same way as the source cloud, samples an amber surface, and rotates both together. The **Primitive benchmark** selector jumps directly among the ten objects, so you can inspect score and visual mismatch without manually locating files.

## How to interpret the result

A low score does not mean the cuboid captures object semantics; it only means its planar surface lies relatively near the source points. A cuboid will tend to look comparatively good for boxy, compact shapes and poorly for cones, rings, thin wings, branches, and disconnected parts. Use the visualizer together with the two Chamfer directions when diagnosing a score:

- Large `P → Q` error usually means excessive empty cuboid surface.
- Large `Q → P` error usually means missing object structure.

The single-cuboid baseline is the floor for the next comparison. `k` cuboids should reduce error by assigning components to separate regions, but can introduce overlap, sensitivity to initialization, and unclear component budgets. Superquadrics, ellipsoids, and medial spheres trade planar faces for smoother surfaces, but each needs an explicit fitting objective and comparable parameter budget.
