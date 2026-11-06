#!/usr/bin/env python3
"""Bounded CPU/CUDA still-frame comparisons; never encode video or retrain models."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def checked(command, **kwargs):
    """Stop on build/execution failure rather than measuring a stale executable."""
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def digest(path):
    """Fingerprint exact executable, library and recorded input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Build a Release driver, run four bounded poses and retain all timing/quality samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/fruit-render-gpu-bench')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'build/fruit-render-gpu-benchmark')
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/results/fruit_render/cuda.json')
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--reuse-samples', action='store_true')
    args = parser.parse_args()
    build, work = args.build_dir.resolve(), args.work_dir.resolve()
    executable = build/'fruit_render_gpu_bench'
    if not args.reuse_samples:
        if not args.no_build:
            checked(['cmake', '-S', 'benchmarks/cad', '-B', str(build), '-DCMAKE_BUILD_TYPE=Release'])
            checked(['cmake', '--build', str(build), '--target', 'fruit_render_gpu_bench', '-j', '4'])
        checked([str(executable), str(ROOT), str(work), str(args.repetitions)])
    quality = json.loads((work/'quality.json').read_text())
    with (work/'samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(quality['frames']) != 4 or len(rows) != 8*quality['repetitions']:
        raise ValueError('Incomplete CPU/CUDA samples')
    stats = {}
    for backend in ['cpu', 'cuda']:
        values = [float(r['milliseconds']) for r in rows if r['backend'] == backend]
        stats[backend] = dict(count=len(values), mean_ms=statistics.mean(values),
                              median_ms=statistics.median(values), min_ms=min(values),
                              max_ms=max(values), stdev_ms=statistics.stdev(values))
    for q in quality['frames']:
        if q['rgb_rmse_255'] > 3 or q['id_mismatch_fraction'] > .001:
            raise ValueError('CUDA visual/geometry gate failed')
    linked = subprocess.check_output(['ldd', str(executable)], text=True)
    library = next(Path(line.split('=>', 1)[1].split()[0])
                   for line in linked.splitlines() if 'libdatamunge' in line and '=>' in line)
    compiler = next(line.split('=', 1)[1] for line in (build/'CMakeCache.txt').read_text().splitlines()
                    if line.startswith('CMAKE_CXX_COMPILER:'))
    paths = ['src/datamunge/cad/render.cpp', 'src/datamunge/cad/detail/cuda_types.hpp',
             'src/datamunge/cad/detail/render_ptx.hpp', 'shaders/cad/render.cu',
             'benchmarks/cad/fruit_render_gpu.cpp', 'benchmarks/cad/fruit_replay.hpp']
    result = dict(native_samples_reused=args.reuse_samples, timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  methodology='Four recorded poses (idle, classified pickup, moving arm, late stream); '
                  'three repeats by default, alternating CPU/CUDA order. Same prepared snapshot, actors, '
                  '960x640 output and full material/shadow/AO settings. Both calls include combined CPU '
                  'BVH work; CUDA also includes uploads, dispatch and synchronous readback. Initial base '
                  'packing/upload/render and renderer initialization are recorded separately. Warm device '
                  'and filesystem caches, unpinned host. No video, neural training, or full-minute replay. '
                  'Sparse pose timings are not an end-to-end video or full-minute estimate.',
                  quality=quality, statistics=stats,
                  median_call_speedup=stats['cpu']['median_ms']/stats['cuda']['median_ms'],
                  raw_samples=rows, executable_sha256=digest(executable), library_sha256=digest(library),
                  linked_libraries=linked, compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                  logical_cpus=os.cpu_count(), source_hashes={p: digest(ROOT/p) for p in paths},
                  input_hashes={p: digest(ROOT/'docs/mepml/examples/build/fruit_stream_realistic'/p)
                                for p in ['trajectory.csv', 'fruit_trajectory.csv']})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('Median complete render calls: CPU %.2f ms; CUDA %.2f ms (%.2fx)' %
          (stats['cpu']['median_ms'], stats['cuda']['median_ms'], result['median_call_speedup']))
    print('Results:', output)


if __name__ == '__main__':
    main()
