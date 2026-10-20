#!/usr/bin/env python3
"""Create the fixed ten-object point-cloud subset for primitive fitting."""
import argparse
import csv
from pathlib import Path

import numpy as np


BENCHMARK = (
    ('01_airplane_08', 'articulated, thin wings'),
    ('04_banana_08', 'curved elongated'),
    ('10_bottle_08', 'rotational'),
    ('15_chair_08', 'multi-part furniture'),
    ('16_Christmas_tree_08', 'branched'),
    ('21_elephant_08', 'organic articulated'),
    ('27_guitar_08', 'curved body and thin neck'),
    ('43_motorcycle_08', 'complex mechanical'),
    ('52_ring_08', 'hollow toroidal'),
    ('61_spider_08', 'many thin limbs'),
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data/neuro-3d')
    parser.add_argument('--participant', default='sub01', choices=[f'sub{i:02d}' for i in range(1, 13)])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    extracted = args.data_dir / 'extracted'
    output_dir = args.output_dir or args.data_dir / 'processed/v1'
    labels = np.load(extracted / args.participant / 'name_label.npy', allow_pickle=False).astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, xyz, rgb, names = [], [], [], []
    for rank, (label, shape_family) in enumerate(BENCHMARK, start=1):
        category_id, object_name = label.split('_', 1)
        category, variant = object_name.rsplit('_', 1)
        geometry = extracted / 'point_cloud' / f'{object_name}.npy'
        trials = np.flatnonzero(labels == label)
        if not geometry.is_file() or len(trials) != 4:
            raise ValueError(f'Expected point cloud and four held-out trials for {label}')
        points = np.load(geometry, allow_pickle=False)
        if points.shape != (8192, 6) or not np.isfinite(points).all():
            raise ValueError(f'Unexpected point cloud for {label}: {points.shape}')
        rows.append({
            'benchmark_rank': rank,
            'label': label,
            'category_id': category_id,
            'category': category,
            'variant': variant,
            'shape_family': shape_family,
            'split_role': 'test',
            'trial_indices': ' '.join(map(str, trials.tolist())),
            'geometry_path': geometry.relative_to(args.data_dir).as_posix(),
            'point_count': len(points),
        })
        names.append(label)
        xyz.append(points[:, :3])
        rgb.append(points[:, 3:])

    csv_path = output_dir / 'primitive_benchmark_10_objects.csv'
    fields = ['benchmark_rank', 'label', 'category_id', 'category', 'variant', 'shape_family', 'split_role', 'trial_indices', 'geometry_path', 'point_count']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    npz_path = output_dir / 'primitive_benchmark_10_objects.npz'
    np.savez_compressed(npz_path, labels=np.array(names), xyz=np.stack(xyz), rgb=np.stack(rgb))
    print(f'Wrote {len(rows)} objects to {csv_path} and {npz_path}')


if __name__ == '__main__':
    main()
