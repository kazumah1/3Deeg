#!/usr/bin/env python3
"""Fit k superquadrics initialized from precomputed k-oriented cuboids.

The benchmark uses a shared 5x5 epsilon sweep, selects candidates by the
full-object symmetric squared Chamfer distance, then refines each component's
two exponents continuously.  Component-local optimization is used only to
make the sweep tractable; all reported scores are always union-to-full-cloud.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from benchmark_oriented_cuboids import chamfer_squared, selected_labels
from benchmark_k_oriented_cuboids import cluster_points


EPSILON_GRID = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
EPSILON_BOUNDS = (0.1, 1.0)
OBJECTIVE_SURFACE_SAMPLES = 384
OBJECTIVE_POINT_SAMPLES = 1024


def signed_power(values, exponent):
    return np.sign(values) * np.abs(values) ** exponent


def angle_samples(count, seed):
    """Deterministic quasi-uniform parameter samples for a superquadric."""
    generator = np.random.default_rng(seed)
    eta = generator.uniform(-np.pi / 2, np.pi / 2, count)
    omega = generator.uniform(-np.pi, np.pi, count)
    return eta, omega


def superquadric_surface(center, axes, scales, epsilon, count, seed):
    eta, omega = angle_samples(count, seed)
    ce, se = np.cos(eta), np.sin(eta)
    co, so = np.cos(omega), np.sin(omega)
    e1, e2 = epsilon
    local = np.column_stack((scales[0] * signed_power(ce, e1) * signed_power(co, e2),
                             scales[1] * signed_power(ce, e1) * signed_power(so, e2),
                             scales[2] * signed_power(se, e1)))
    return center + local @ axes.T


def estimated_area(center, axes, scales, epsilon, seed):
    """Estimate relative area by triangulating a fixed parameter grid."""
    eta = np.linspace(-np.pi / 2 + .01, np.pi / 2 - .01, 18)
    omega = np.linspace(-np.pi, np.pi, 28, endpoint=False)
    eta, omega = np.meshgrid(eta, omega, indexing='ij')
    e1, e2 = epsilon
    local = np.stack((scales[0] * signed_power(np.cos(eta), e1) * signed_power(np.cos(omega), e2),
                      scales[1] * signed_power(np.cos(eta), e1) * signed_power(np.sin(omega), e2),
                      scales[2] * signed_power(np.sin(eta), e1)), axis=-1)
    points = center + local @ axes.T
    area = 0.0
    for row in range(points.shape[0] - 1):
        for column in range(points.shape[1]):
            nxt = (column + 1) % points.shape[1]
            a, b, c, d = points[row, column], points[row + 1, column], points[row + 1, nxt], points[row, nxt]
            area += np.linalg.norm(np.cross(b - a, c - a)) / 2 + np.linalg.norm(np.cross(c - a, d - a)) / 2
    return area


def local_points(points, count, seed):
    if len(points) <= count:
        return points
    return points[np.random.default_rng(seed).choice(len(points), count, replace=False)]


def unpack(parameters, initial_center, initial_axes, initial_scales, include_epsilon, epsilon):
    center = initial_center + parameters[:3]
    scales = initial_scales * np.exp(parameters[3:6])
    axes = initial_axes @ Rotation.from_rotvec(parameters[6:9]).as_matrix()
    return center, axes, scales, (parameters[9:11] if include_epsilon else epsilon)


def optimize_component(initial, target, epsilon, seed, refine_epsilon=False):
    """Bounded local pose/scale optimization, with optional epsilon refinement."""
    center, axes, scales = initial
    target = local_points(target, OBJECTIVE_POINT_SAMPLES, seed)
    initial_parameters = np.zeros(11 if refine_epsilon else 9)
    if refine_epsilon:
        initial_parameters[9:11] = epsilon
    diameter = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    bounds = [(-diameter * .25, diameter * .25)] * 3 + [(-1.2, .8)] * 3 + [(-.45, .45)] * 3
    if refine_epsilon:
        bounds += [EPSILON_BOUNDS, EPSILON_BOUNDS]

    def objective(parameters):
        fitted_center, fitted_axes, fitted_scales, fitted_epsilon = unpack(
            parameters, center, axes, scales, refine_epsilon, epsilon)
        surface = superquadric_surface(fitted_center, fitted_axes, fitted_scales, fitted_epsilon,
                                       OBJECTIVE_SURFACE_SAMPLES, seed)
        return chamfer_squared(surface, target)

    result = minimize(objective, initial_parameters, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 12 if refine_epsilon else 8, 'ftol': 1e-5})
    fitted = unpack(result.x, center, axes, scales, refine_epsilon, epsilon)
    return fitted, int(result.nit), 'converged' if result.success else 'max_iterations'


def load_initial_components(root, k, label):
    fitted = np.load(root / f'k{k}_ten_object/fitted_cuboids.npz', allow_pickle=False)
    labels = fitted['labels'].astype(str)
    index = int(np.flatnonzero(labels == label)[0])
    return list(zip(fitted['centers'][index], fitted['axes'][index], fitted['extents'][index] / 2))


def sample_union(components, total, seed):
    areas = np.array([estimated_area(*component, seed + index) for index, component in enumerate(components)])
    raw_counts = areas / areas.sum() * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - counts.sum()
    if remainder:
        counts[np.argsort(raw_counts - counts)[-remainder:]] += 1
    return np.concatenate([superquadric_surface(*component, int(count), seed + index)
                           for index, (component, count) in enumerate(zip(components, counts))])


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data/neuro-3d')
    parser.add_argument('--participant', choices=[f'sub{i:02d}' for i in range(1, 13)], default='sub01')
    parser.add_argument('--manifest', type=Path, default=root / 'benchmark/primitive_10_objects.csv')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    if not 1 <= args.k <= args.samples:
        raise ValueError('--k must be at least 1 and no greater than --samples')
    extracted = args.data_dir / 'extracted'
    labels = np.load(extracted / args.participant / 'name_label.npy', allow_pickle=False).astype(str)
    objects = selected_labels(labels, args.manifest)
    cuboid_root = root / 'benchmark/k_oriented_cuboid'
    if not (cuboid_root / f'k{args.k}_ten_object/fitted_cuboids.npz').is_file():
        raise FileNotFoundError(f'Precompute k={args.k} cuboids before fitting superquadrics')
    output_dir = args.output_dir or root / 'benchmark/superquadric' / f'k{args.k}_ten_object'
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, centers_all, axes_all, scales_all, epsilons_all = [], [], [], [], []
    for object_index, label in enumerate(objects):
        category_id, name = label.split('_', 1)
        category, variant = name.rsplit('_', 1)
        xyz = np.load(extracted / 'point_cloud' / f'{name}.npy', allow_pickle=False)[:, :3].astype(np.float64, copy=False)
        assignment, _, _ = cluster_points(xyz, args.k)
        initial = load_initial_components(cuboid_root, args.k, label)
        clusters = [xyz[assignment == component] for component in range(args.k)]
        started = perf_counter()
        best = None
        total_iterations = 0
        for e1 in EPSILON_GRID:
            for e2 in EPSILON_GRID:
                components, iterations, statuses = [], 0, []
                for component, (initial_component, cluster) in enumerate(zip(initial, clusters)):
                    fitted, used, status = optimize_component(initial_component, cluster, (e1, e2),
                                                              args.seed + object_index * 10000 + component * 100 + int(e1 * 10) * 10 + int(e2 * 10))
                    components.append(fitted); iterations += used; statuses.append(status)
                score = chamfer_squared(sample_union(components, args.samples, args.seed + object_index * 1000), xyz)
                if best is None or score < best[0]:
                    best = (score, (e1, e2), components, iterations, statuses)
        _, shared_epsilon, components, sweep_iterations, statuses = best
        refined, refine_iterations, refine_statuses = [], 0, []
        for component, (fitted_component, cluster) in enumerate(zip(components, clusters)):
            fitted, used, status = optimize_component(fitted_component[:3], cluster, shared_epsilon,
                                                      args.seed + object_index * 10000 + 9000 + component, refine_epsilon=True)
            refined.append(fitted); refine_iterations += used; refine_statuses.append(status)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        score = chamfer_squared(sample_union(refined, args.samples, args.seed + object_index * 1000), xyz)
        evaluation_seconds = perf_counter() - started
        centers, axes, scales, epsilons = (np.stack(values) for values in zip(*refined))
        diagonal_squared = float(np.sum((xyz.max(axis=0) - xyz.min(axis=0)) ** 2))
        rows.append({'label': label, 'category_id': category_id, 'category': category, 'variant': variant, 'k': args.k,
                     'point_count': len(xyz), 'surface_samples': args.samples, 'fit_seconds': f'{fit_seconds:.9f}',
                     'evaluation_seconds': f'{evaluation_seconds:.9f}', 'chamfer_squared': f'{score:.9f}',
                     'chamfer_normalized_by_cloud_diagonal_squared': f'{score / diagonal_squared:.9f}',
                     'shared_epsilon_1': f'{shared_epsilon[0]:.3f}', 'shared_epsilon_2': f'{shared_epsilon[1]:.3f}',
                     'final_epsilons': json.dumps(epsilons.tolist()), 'convergence_status': ','.join(refine_statuses),
                     'iterations': sweep_iterations + refine_iterations})
        centers_all.append(centers); axes_all.append(axes); scales_all.append(scales); epsilons_all.append(epsilons)
    with (output_dir / 'per_object.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output_dir / 'fitted_superquadrics.npz', labels=np.array(objects), centers=np.stack(centers_all),
                        axes=np.stack(axes_all), scales=np.stack(scales_all), epsilons=np.stack(epsilons_all))
    scores = np.array([float(row['chamfer_squared']) for row in rows])
    summary = {'method': 'k_superquadrics_shared_sweep_then_per_component_epsilon_refinement', 'k': args.k,
               'participant': args.participant, 'objects': len(rows), 'surface_samples_per_object': args.samples,
               'shared_epsilon_grid': EPSILON_GRID.tolist(), 'epsilon_bounds': list(EPSILON_BOUNDS),
               'initialization': 'matching precomputed k-oriented cuboids: center, PCA axes, half extents',
               'objective': 'symmetric squared Chamfer; local objectives for optimization, full union for selection and reporting',
               'mean_chamfer_squared': float(scores.mean()), 'median_chamfer_squared': float(np.median(scores)),
               'total_fit_seconds': float(sum(float(row['fit_seconds']) for row in rows)),
               'total_evaluation_seconds': float(sum(float(row['evaluation_seconds']) for row in rows))}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
