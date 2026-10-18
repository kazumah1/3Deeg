#!/usr/bin/env python3
"""Serve a local EEG/point-cloud explorer. No source data is modified."""
import argparse
from collections import Counter
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


def load_cuboid_fit(path, kind):
    if not path.is_file():
        return {}
    fitted = np.load(path, allow_pickle=False)
    metrics = {}
    metric_path = path.with_name('per_object.csv')
    if metric_path.is_file():
        with metric_path.open(newline='', encoding='utf-8') as handle:
            metrics = {row['label']: dict(chamfer_squared=float(row['chamfer_squared'])) for row in csv.DictReader(handle)}
    result = {}
    for index, label in enumerate(fitted['labels']):
        centers, axes, extents = fitted['centers'][index], fitted['axes'][index], fitted['extents'][index]
        if centers.ndim == 1:
            centers, axes, extents = centers[None], axes[None], extents[None]
        result[str(label)] = dict(kind=kind, k=len(centers), components=[dict(center=center.tolist(), axes=axis.tolist(), extents=extent.tolist())
                                                                         for center, axis, extent in zip(centers, axes, extents)],
                                  **metrics.get(str(label), {}))
    return result


class Dataset:
    def __init__(self, root, participant):
        self.root = root / 'extracted'
        self.participant = participant
        self.labels = np.load(self.root / participant / 'name_label.npy', allow_pickle=False)
        self.eeg = {window: np.load(self.root / participant / f'process_data_{window}.npy', mmap_mode='r', allow_pickle=False)
                    for window in ['1s_250Hz', '6s_100Hz']}
        self.objects = []
        for label in sorted(set(self.labels.tolist())):
            category_id, name = label.split('_', 1)
            category, variant = name.rsplit('_', 1)
            path = self.root / 'point_cloud' / f'{name}.npy'
            if not category_id.isdigit() or not path.is_file():
                raise ValueError(f'Cannot map label to geometry: {label}')
            self.objects.append(dict(label=label, name=name, category=category, variant=variant,
                                     trials=np.flatnonzero(self.labels == label).tolist()))
        for window, data in self.eeg.items():
            if data.shape != (len(self.labels), 64, 250 if window == '1s_250Hz' else 600):
                raise ValueError(f'Unexpected EEG shape: {window}: {data.shape}')
        benchmark_root = Path(__file__).resolve().parents[1] / 'benchmark'
        self.cuboid_fits = {
            'single': load_cuboid_fit(benchmark_root / 'oriented_cuboid/ten_object/fitted_cuboids.npz', 'single'),
            'k4': load_cuboid_fit(benchmark_root / 'k_oriented_cuboid/k4_ten_object/fitted_cuboids.npz', 'k4'),
        }
        self.cuboid_fits = {kind: fits for kind, fits in self.cuboid_fits.items() if fits}

    def summary(self):
        return dict(participant=self.participant, trials=len(self.labels), objects=self.objects,
                    categories=len({o['category'] for o in self.objects}),
                    repeats=dict(Counter(len(o['trials']) for o in self.objects)),
                    benchmark_objects=sorted(set().union(*[set(fits) for fits in self.cuboid_fits.values()]) if self.cuboid_fits else set()),
                    available_cuboid_fits=list(self.cuboid_fits))

    def sample(self, trial, window):
        if window not in self.eeg or not 0 <= trial < len(self.labels):
            raise ValueError('Invalid trial or recording window')
        label = str(self.labels[trial])
        points = np.load(self.root / 'point_cloud' / f'{label.split("_", 1)[1]}.npy', allow_pickle=False)
        eeg = np.asarray(self.eeg[window][trial])
        if points.shape != (8192, 6) or not np.isfinite(points).all() or not np.isfinite(eeg).all():
            raise ValueError('Unexpected point-cloud shape or nonfinite data')
        return dict(label=label, trial=trial, window=window, hz=250 if window == '1s_250Hz' else 100,
                    points=points.tolist(), eeg=eeg.tolist(), std=float(eeg.std()))

    def cuboid(self, label, kind):
        if kind not in self.cuboid_fits or label not in self.cuboid_fits[kind]:
            raise ValueError('No fitted cuboid is available for this object and fit type')
        return dict(label=label, **self.cuboid_fits[kind][label])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data/neuro-3d')
    parser.add_argument('--participant', choices=[f'sub{i:02d}' for i in range(1, 13)], default='sub01')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    dataset = Dataset(args.data_dir, args.participant)
    html = (Path(__file__).resolve().parents[1] / 'explorer/index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            try:
                if route.path == '/':
                    payload, mime = html, 'text/html; charset=utf-8'
                elif route.path == '/api/summary':
                    payload, mime = json.dumps(dataset.summary()).encode(), 'application/json'
                elif route.path == '/api/sample':
                    query = parse_qs(route.query)
                    sample = dataset.sample(int(query.get('trial', ['0'])[0]), query.get('window', ['1s_250Hz'])[0])
                    payload, mime = json.dumps(sample, allow_nan=False).encode(), 'application/json'
                elif route.path == '/api/cuboid':
                    query = parse_qs(route.query)
                    payload, mime = json.dumps(dataset.cuboid(query.get('label', [''])[0], query.get('kind', ['single'])[0]), allow_nan=False).encode(), 'application/json'
                else:
                    self.send_error(404)
                    return
            except (ValueError, OSError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Explorer: http://127.0.0.1:{args.port} — {args.participant}, {len(dataset.objects)} objects. Ctrl+C stops it.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
