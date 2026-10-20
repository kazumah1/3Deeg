#!/usr/bin/env python3
"""Fit PCA-aligned enclosing cuboids and score them with squared Chamfer distance."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree


def fit_oriented_cuboid(xyz):
    center = xyz.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((xyz - center).T))
    axes = axes[:, ::-1]
    local = (xyz - center) @ axes
    lower, upper = local.min(axis=0), local.max(axis=0)
    return center + ((lower + upper) / 2) @ axes.T, axes, upper - lower


def sample_surface(center, axes, extents, count, generator):
    half = extents / 2
    areas = np.array([extents[1] * extents[2], extents[1] * extents[2], extents[0] * extents[2],
                      extents[0] * extents[2], extents[0] * extents[1], extents[0] * extents[1]])
    faces = generator.choice(6, count, p=areas / areas.sum())
    local = generator.uniform(-half, half, (count, 3))
    for axis in range(3):
        local[faces == 2 * axis, axis] = -half[axis]
        local[faces == 2 * axis + 1, axis] = half[axis]
    return center + local @ axes.T


def chamfer_squared(first, second):
    first_to_second = cKDTree(second).query(first, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, workers=-1)[0]
    return float(np.mean(first_to_second ** 2) + np.mean(second_to_first ** 2))


def selected_labels(labels, manifest):
    if manifest is None:
        return sorted(label for label in set(labels) if label.rsplit('_', 1)[1] in {f'{i:02d}' for i in range(8)})
    with manifest.open(newline='', encoding='utf-8') as handle:
        result = [row['label'] for row in csv.DictReader(handle)]
    if len(result) != len(set(result)) or not result:
        raise ValueError('Manifest must contain a nonempty, unique label column')
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data/neuro-3d')
    parser.add_argument('--participant', choices=[f'sub{i:02d}' for i in range(1, 13)], default='sub01')
    parser.add_argument('--manifest', type=Path, help='CSV with a label column; omitting it uses all 576 training objects.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError('--samples must be positive')
    extracted = args.data_dir / 'extracted'
    labels = np.load(extracted / args.participant / 'name_label.npy', allow_pickle=False).astype(str)
    objects = selected_labels(labels, args.manifest)
    if any(label not in labels for label in objects):
        raise ValueError('Manifest contains a label unavailable for this participant')
    output_dir = args.output_dir or args.data_dir / 'processed/v1' / ('oriented_cuboid_manifest_benchmark' if args.manifest else 'oriented_cuboid_train_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, centers, axes_all, extents_all = [], [], [], []
    for index, label in enumerate(objects):
        category_id, name = label.split('_', 1)
        category, variant = name.rsplit('_', 1)
        xyz = np.load(extracted / 'point_cloud' / f'{name}.npy', allow_pickle=False)[:, :3].astype(np.float64, copy=False)
        started = perf_counter(); center, axes, extents = fit_oriented_cuboid(xyz); fit_seconds = perf_counter() - started
        started = perf_counter(); score = chamfer_squared(sample_surface(center, axes, extents, args.samples, np.random.default_rng(args.seed + index)), xyz); evaluation_seconds = perf_counter() - started
        diagonal_squared = float(np.sum(extents ** 2))
        rows.append({'label': label, 'category_id': category_id, 'category': category, 'variant': variant,
                     'point_count': len(xyz), 'surface_samples': args.samples, 'fit_seconds': f'{fit_seconds:.9f}',
                     'evaluation_seconds': f'{evaluation_seconds:.9f}', 'chamfer_squared': f'{score:.9f}',
                     'chamfer_normalized_by_cuboid_diagonal_squared': f'{score / diagonal_squared:.9f}',
                     'convergence_status': 'closed_form_pca', 'iterations': 1})
        centers.append(center); axes_all.append(axes); extents_all.append(extents)
    with (output_dir / 'per_object.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output_dir / 'fitted_cuboids.npz', labels=np.array(objects), centers=np.stack(centers), axes=np.stack(axes_all), extents=np.stack(extents_all))
    scores = np.array([float(row['chamfer_squared']) for row in rows])
    summary = {'method': 'one_pca_aligned_oriented_bounding_cuboid', 'participant': args.participant, 'objects': len(rows),
               'points_per_object': 8192, 'surface_samples_per_object': args.samples, 'seed': args.seed,
               'manifest': str(args.manifest) if args.manifest else None, 'convergence_status': 'closed_form_pca',
               'iterations_per_object': 1, 'chamfer_definition': 'symmetric mean squared nearest-neighbor distance',
               'mean_chamfer_squared': float(scores.mean()), 'median_chamfer_squared': float(np.median(scores)),
               'total_fit_seconds': float(sum(float(row['fit_seconds']) for row in rows)),
               'total_evaluation_seconds': float(sum(float(row['evaluation_seconds']) for row in rows))}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
