#!/usr/bin/env python3
"""Fit four deterministic PCA-oriented cuboids and score their union."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_oriented_cuboids import chamfer_squared, sample_surface, selected_labels


K = 4
MAX_ITERATIONS = 100
TOLERANCE = 1e-8


def fit_oriented_cuboid(xyz):
    center = xyz.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((xyz - center).T))
    axes = axes[:, ::-1]
    local = (xyz - center) @ axes
    lower, upper = local.min(axis=0), local.max(axis=0)
    return center + ((lower + upper) / 2) @ axes.T, axes, upper - lower


def farthest_point_initialization(xyz):
    centers = [xyz[np.argmax(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1))]]
    nearest = np.sum((xyz - centers[0]) ** 2, axis=1)
    for _ in range(1, K):
        center = xyz[np.argmax(nearest)]
        centers.append(center)
        nearest = np.minimum(nearest, np.sum((xyz - center) ** 2, axis=1))
    return np.stack(centers)


def cluster_points(xyz):
    """Run deterministic Lloyd k-means and return assignments and convergence metadata."""
    centers = farthest_point_initialization(xyz)
    for iteration in range(1, MAX_ITERATIONS + 1):
        distances = np.sum((xyz[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignment = np.argmin(distances, axis=1)
        updated = np.empty_like(centers)
        for component in range(K):
            members = xyz[assignment == component]
            if len(members) == 0:
                updated[component] = xyz[np.argmax(np.min(distances, axis=1))]
            else:
                updated[component] = members.mean(axis=0)
        shift = np.max(np.linalg.norm(updated - centers, axis=1))
        centers = updated
        if shift <= TOLERANCE:
            return assignment, iteration, 'converged'
    return assignment, MAX_ITERATIONS, 'max_iterations'


def component_sample_counts(extents, total):
    areas = np.array([2 * (extent[0] * extent[1] + extent[0] * extent[2] + extent[1] * extent[2]) for extent in extents])
    raw = areas / areas.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    if remainder:
        counts[np.argsort(raw - counts)[-remainder:]] += 1
    return counts


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data/neuro-3d')
    parser.add_argument('--participant', choices=[f'sub{i:02d}' for i in range(1, 13)], default='sub01')
    parser.add_argument('--manifest', type=Path, help='CSV with a label column; omitting it uses all 576 training objects.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261017)
    args = parser.parse_args()
    if args.samples < K:
        raise ValueError(f'--samples must be at least {K}')
    extracted = args.data_dir / 'extracted'
    labels = np.load(extracted / args.participant / 'name_label.npy', allow_pickle=False).astype(str)
    objects = selected_labels(labels, args.manifest)
    if any(label not in labels for label in objects):
        raise ValueError('Manifest contains a label unavailable for this participant')
    output_dir = args.output_dir or root / 'benchmark/k_oriented_cuboid' / ('k4_ten_object' if args.manifest else 'k4_training')
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, centers_all, axes_all, extents_all = [], [], [], []
    for index, label in enumerate(objects):
        category_id, name = label.split('_', 1)
        category, variant = name.rsplit('_', 1)
        xyz = np.load(extracted / 'point_cloud' / f'{name}.npy', allow_pickle=False)[:, :3].astype(np.float64, copy=False)
        started = perf_counter()
        assignment, iterations, status = cluster_points(xyz)
        components = [fit_oriented_cuboid(xyz[assignment == component]) for component in range(K)]
        fit_seconds = perf_counter() - started
        centers, axes, extents = (np.stack(values) for values in zip(*components))
        counts = component_sample_counts(extents, args.samples)
        started = perf_counter()
        surfaces = [sample_surface(center, axis, extent, count, np.random.default_rng(args.seed + index * K + component))
                    for component, (center, axis, extent, count) in enumerate(zip(centers, axes, extents, counts))]
        score = chamfer_squared(np.concatenate(surfaces), xyz)
        evaluation_seconds = perf_counter() - started
        union_diagonal_squared = float(np.sum((xyz.max(axis=0) - xyz.min(axis=0)) ** 2))
        rows.append({'label': label, 'category_id': category_id, 'category': category, 'variant': variant, 'k': K,
                     'point_count': len(xyz), 'surface_samples': args.samples, 'fit_seconds': f'{fit_seconds:.9f}',
                     'evaluation_seconds': f'{evaluation_seconds:.9f}', 'chamfer_squared': f'{score:.9f}',
                     'chamfer_normalized_by_cloud_diagonal_squared': f'{score / union_diagonal_squared:.9f}',
                     'convergence_status': status, 'iterations': iterations})
        centers_all.append(centers); axes_all.append(axes); extents_all.append(extents)

    with (output_dir / 'per_object.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output_dir / 'fitted_cuboids.npz', labels=np.array(objects), centers=np.stack(centers_all),
                        axes=np.stack(axes_all), extents=np.stack(extents_all))
    scores = np.array([float(row['chamfer_squared']) for row in rows])
    summary = {'method': 'k_pca_aligned_oriented_bounding_cuboids', 'k': K, 'participant': args.participant,
               'objects': len(rows), 'points_per_object': 8192, 'surface_samples_per_object': args.samples,
               'seed': args.seed, 'manifest': str(args.manifest) if args.manifest else None,
               'clustering': 'deterministic farthest-point initialization followed by Lloyd k-means',
               'max_iterations': MAX_ITERATIONS, 'tolerance': TOLERANCE,
               'chamfer_definition': 'symmetric mean squared nearest-neighbor distance',
               'mean_chamfer_squared': float(scores.mean()), 'median_chamfer_squared': float(np.median(scores)),
               'total_fit_seconds': float(sum(float(row['fit_seconds']) for row in rows)),
               'total_evaluation_seconds': float(sum(float(row['evaluation_seconds']) for row in rows))}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
