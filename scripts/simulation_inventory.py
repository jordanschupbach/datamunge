#!/usr/bin/env python3
"""Freeze source provenance and the numerical API inventory for simulation/GPU parity.

Reads Doxygen XML when supplied and hashes the actual working-tree inputs rather
than attributing uncommitted contents to HEAD. No backend is marked implemented
or tested by this source inventory. Output is deterministic for identical inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    """Return SHA-256 of the exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root, patterns):
    """Record sorted relative paths and content hashes, excluding build outputs."""
    files = {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}


def operations(xml):
    """Collect deduplicated public NN declarations from the complete XML inventory."""
    found = {}
    for path in sorted(xml.glob('*.xml')):
        compound = ET.parse(path).getroot().find('compounddef')
        if compound is None:
            continue
        namespace = compound.findtext('compoundname', '')
        for member in compound.findall('.//memberdef'):
            location = member.find('location')
            if location is None or member.get('prot', 'public') != 'public':
                continue
            source = location.get('file', '')
            if 'include/datamunge/nn/' not in source or member.get('kind') != 'function':
                continue
            name = member.findtext('qualifiedname') or namespace + '::' + member.findtext('name', '')
            signature = name + member.findtext('argsstring', '')
            found[signature] = {'signature': signature, 'source': source[source.index('include/'):],
                                'cuda': 'pending', 'vulkan': 'pending'}
    return [found[k] for k in sorted(found)]


def main():
    """Write an explicit, reviewable provenance and parity baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vsim', type=Path, required=True)
    parser.add_argument('--xml', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = args.vsim.resolve()
    if not (reference / 'include/vsim/physics.hpp').is_file():
        parser.error('--vsim must name the original engine checkout')
    revision = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    baseline = {'schema': 1, 'vsim_revision': revision,
                'vsim_files': snapshot(reference, ['include/vsim/*.hpp', 'src/*.cpp', 'tests/*.cpp',
                                                  'docs/*.md', 'examples/*/scene.json', 'examples/*/*.lua',
                                                  'shaders/*', 'scenes/*.json', 'scenes/*.lua']),
                'nn_files': snapshot(ROOT, ['include/datamunge/nn/*.hpp', 'src/datamunge/nn/*.cpp']),
                'nn_operations': operations(args.xml) if args.xml else [],
                'required_backends': ['cpu', 'cuda', 'vulkan'],
                'dtypes': ['bool', 'int32', 'int64', 'float32', 'float64']}
    if args.xml and not baseline['nn_operations']:
        parser.error('XML contains no public NN operations; regenerate Doxygen')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(baseline, indent=2) + '\n')
    print(f"Recorded {len(baseline['vsim_files'])} reference files, {len(baseline['nn_files'])} NN files, "
          f"{len(baseline['nn_operations'])} public declarations")


if __name__ == '__main__':
    main()
