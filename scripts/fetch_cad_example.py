#!/usr/bin/env python3
"""Fetch the pinned Seeed robot-arm example assets and verify their hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'build/cad-assets/rebot')
    parser.add_argument('--check', action='store_true', help='Verify cached assets without network access')
    args = parser.parse_args()
    manifest = json.loads((root / 'examples/cad/rebot-assets.json').read_text())
    prefix = 'https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/' + manifest['revision'] + '/'

    def fetch(entry):
        target = args.output / entry['path']
        if target.is_file() and target.stat().st_size == entry['size']:
            if hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256']:
                return
        if args.check:
            raise RuntimeError('Missing or changed asset: ' + str(target))
        url = prefix + urllib.parse.quote(entry['path'], safe='/')
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if len(data) != entry['size'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise RuntimeError('Asset hash mismatch: ' + entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.download')
        temporary.write_bytes(data)
        temporary.replace(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, manifest['files']))
    print(f"Verified {len(manifest['files'])} pinned DevArm assets in {args.output}")


if __name__ == '__main__':
    main()
