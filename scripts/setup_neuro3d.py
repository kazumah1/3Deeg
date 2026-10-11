#!/usr/bin/env python3
"""Selective Neuro-3D setup using Python's standard library and curl."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import zlib

REPO = 'guozq21/neuro-3D'
REVISION = '31ebd72706a6b6b40bfe77022f9ca48a7a12dfdd'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        raise ValueError(f'Missing or incorrect size: {path}')
    if digest(path) != entry['lfs']['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}; remove it and download again')


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename:
                raise ValueError(f'Unsafe archive member: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Symlink archive member: {member.filename}')
        print(f'{path.name}: {len(members)} members, {sum(m.file_size for m in members):,} extracted bytes')
        for member in members[:12]:
            print(f'  {member.filename} ({member.file_size:,} bytes)')
        return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'inspect', 'extract', 'verify'])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--participants', nargs='+', default=['sub01'], choices=[f'sub{i:02d}' for i in range(1, 13)])
    selection.add_argument('--all-participants', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=Path('data/neuro-3d'))
    parser.add_argument('--revision', default=REVISION, help='Hub revision (default: pinned commit)')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    manifest_path = root / 'manifest.json'
    if args.action == 'download':
        if not shutil.which('curl'):
            raise ValueError('curl is required for resumable downloads')
        with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/{args.revision}?blobs=true', timeout=60) as response:
            manifest = json.load(response)
        if manifest_path.exists() and json.loads(manifest_path.read_text())['sha'] != manifest['sha']:
            raise ValueError('Destination contains a different revision; use another data directory')
        root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads(manifest_path.read_text())
    participants = [f'sub{i:02d}' for i in range(1, 13)] if args.all_participants else args.participants
    names = ['point_cloud.zip'] + [f'EEGdata/{p}.zip' for p in participants]
    entries = {entry['rfilename']: entry for entry in manifest['siblings']}
    for name in names:
        entry = entries[name]
        path = root / 'archives' / name
        if args.action == 'download':
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                partial = path.with_suffix(path.suffix + '.part')
                remaining = max(0, entry['size'] - (partial.stat().st_size if partial.exists() else 0))
                if shutil.disk_usage(root).free < remaining:
                    raise ValueError(f'Insufficient disk space for {name}')
                if not partial.exists() or partial.stat().st_size != entry['size']:
                    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--continue-at', '-', '--output', str(partial), f'https://huggingface.co/datasets/{REPO}/resolve/{manifest["sha"]}/{name}'], check=True)
                verify(partial, entry)
                partial.replace(path)
            verify(path, entry)
            print(f'Verified {name}', flush=True)
        elif args.action == 'inspect':
            inspect(path)
        elif args.action == 'verify':
            verify(path, entry)
            members = inspect(path)
            destination = root / 'extracted'
            if destination.exists():
                for member in members:
                    target = destination / member.filename
                    if not target.exists() or (not member.is_dir() and target.stat().st_size != member.file_size):
                        raise ValueError(f'Missing or incorrect extracted file: {target}')
                    if not member.is_dir():
                        crc = 0
                        with target.open('rb') as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b''):
                                crc = zlib.crc32(block, crc)
                        if crc != member.CRC:
                            raise ValueError(f'Extracted CRC mismatch: {target}')
            print(f'Verified {name}', flush=True)
        else:
            verify(path, entry)
            members = inspect(path)
            destination = root / 'extracted'
            destination.mkdir(exist_ok=True)
            if shutil.disk_usage(root).free < sum(m.file_size for m in members):
                raise ValueError(f'Insufficient extraction space for {name}')
            for member in members:
                target = destination / member.filename
                if target.is_symlink() or any(p.is_symlink() for p in target.parents):
                    raise ValueError(f'Symlink in extraction destination: {target}')
            with zipfile.ZipFile(path) as archive:
                archive.extractall(destination)
            print(f'Extracted {name} to {destination}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        sys.exit(f'Error: {error}')
