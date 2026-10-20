#!/usr/bin/env python3
"""Build a trial-to-object CSV index from an extracted Neuro-3D participant."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data/neuro-3d')
    parser.add_argument('--participant', default='sub01', choices=[f'sub{i:02d}' for i in range(1, 13)])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    extracted = args.data_dir / 'extracted'
    labels_path = extracted / args.participant / 'name_label.npy'
    labels = np.load(labels_path, allow_pickle=False)
    if labels.ndim != 1:
        raise ValueError(f'Expected one-dimensional labels, found {labels.shape}: {labels_path}')

    output = args.output or args.data_dir / 'processed/v1' / f'{args.participant}_trial_to_object_index.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    repeats = defaultdict(int)
    rows = []
    for trial, raw_label in enumerate(labels):
        label = str(raw_label)
        category_id, object_name = label.split('_', 1)
        category, variant = object_name.rsplit('_', 1)
        geometry = extracted / 'point_cloud' / f'{object_name}.npy'
        if not category_id.isdigit() or not geometry.is_file():
            raise ValueError(f'Label cannot be mapped to a point cloud: {label}')
        repeats[label] += 1
        rows.append({
            'participant': args.participant,
            'trial': trial,
            'label': label,
            'category_id': category_id,
            'category': category,
            'variant': variant,
            'repeat': repeats[label],
            'split_role': 'test' if variant in {'08', '09'} else 'train',
            'geometry_path': geometry.relative_to(args.data_dir).as_posix(),
        })

    fields = ['participant', 'trial', 'label', 'category_id', 'category', 'variant', 'repeat', 'split_role', 'geometry_path']
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} trials for {len(repeats)} objects to {output}')


if __name__ == '__main__':
    main()
