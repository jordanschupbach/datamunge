#!/usr/bin/env python3
"""Bounded CPU wrist comparisons with exact RGB/depth/ID/crop/box validation."""
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


def digest(path):
    """Fingerprint exact input, code, executable or output bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(command, **kwargs):
    """Stop on build or measurement failure instead of using a stale executable."""
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def main():
    """Measure four poses serially; an optional old library enables paired exact-output checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, choices=range(1, 4), default=3)
    parser.add_argument('--paired-rounds', type=int, choices=range(1, 4), default=2)
    parser.add_argument('--reference-library', type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/fruit-wrist-bench')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'build/fruit-wrist-benchmark')
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/results/fruit_render/wrist.json')
    parser.add_argument('--no-build', action='store_true')
    args = parser.parse_args()
    build, work = args.build_dir.resolve(), args.work_dir.resolve()
    if not args.no_build:
        checked(['cmake', '-S', 'benchmarks/cad', '-B', str(build), '-DCMAKE_BUILD_TYPE=Release'])
        checked(['cmake', '--build', str(build), '--target', 'fruit_wrist_bench', '-j', '4'])
    executable = build/'fruit_wrist_bench'
    reference = args.reference_library.resolve() if args.reference_library else None
    if reference and not reference.is_file():
        parser.error('Reference shared library does not exist')
    runs, samples, golden = [], {}, None
    rounds = args.paired_rounds if reference else 1
    for number in range(rounds):
        order = (['reference', 'candidate'] if number % 2 == 0 else ['candidate', 'reference']) if reference else ['candidate']
        for variant in order:
            destination = work/f'{variant}-{number+1}'
            environment = os.environ.copy()
            environment.pop('LD_LIBRARY_PATH', None)
            if variant == 'reference':
                environment['LD_LIBRARY_PATH'] = str(reference.parent)
            checked([str(executable), str(ROOT), str(destination), str(args.repetitions)], env=environment)
            with (destination/'samples.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 4*args.repetitions+2:
                raise ValueError('Incomplete wrist benchmark')
            hashes = {p.name: digest(p) for p in sorted(destination.iterdir()) if p.suffix in ['.rgba', '.depth', '.ids'] or p.name == 'boxes.csv'}
            if golden is None:
                golden = hashes
            if hashes != golden:
                raise ValueError('Wrist RGB/depth/IDs/crops/boxes differ across implementations or runs')
            runs.append(dict(variant=variant, round=number+1, raw_samples=rows, output_hashes=hashes, load_after=Path('/proc/loadavg').read_text().strip()))
            for row in rows:
                for key in [row['mode'], row['mode']+'_'+row['frame']]:
                    samples.setdefault((variant, key), []).append(float(row['milliseconds']))
    stats = {}
    for (variant, mode), values in samples.items():
        stats.setdefault(variant, {})[mode] = dict(count=len(values), mean_ms=statistics.mean(values),
             median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
             stdev_ms=statistics.stdev(values) if len(values)>1 else 0)
    linked = subprocess.check_output(['ldd', str(executable)], text=True)
    library = next(Path(line.split('=>', 1)[1].split()[0]) for line in linked.splitlines() if 'libdatamunge' in line and '=>' in line)
    compiler = next(line.split('=', 1)[1] for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith('CMAKE_CXX_COMPILER:'))
    source_files = ['src/datamunge/sim/devarm_camera.cpp', 'include/datamunge/sim/devarm_camera.hpp', 'benchmarks/cad/fruit_wrist.cpp', 'benchmarks/cad/fruit_replay.hpp']
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
       methodology='Four recorded poses, 256x256 CPU wrist RGB, full-quality materials/shadows/AO, '
       '16 total render workers. Each timed video-style capture/inspection is primed by the immediate '
       'preceding frame using the same capture/inspection choice as the report. Off-image focus is '
       'a separate control. Paired runs alternate variant order, without overlapping builds or tests. '
       'All native RGB/depth/ID/crop/box outputs are compared exactly. Warm assets, unpinned host; '
       'equal-weight sparse cases are not a full-video or full-minute estimate. No video, encoding or training.',
       statistics=stats, runs=runs, output_files=len(golden), matching_output_files=len(golden) if reference else None,
       reference_comparison_performed=reference is not None, exact_outputs_match=True if reference else None,
       compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0], linked_libraries=linked,
       executable_sha256=digest(executable), candidate_library_sha256=digest(library),
       reference_library_sha256=digest(reference) if reference else None,
       reference_source_sha256=digest(reference.parent/'devarm_camera.cpp') if reference and (reference.parent/'devarm_camera.cpp').is_file() else None,
       source_hashes={p: digest(ROOT/p) for p in source_files},
       input_hashes={p: digest(ROOT/'docs/mepml/examples/build/fruit_stream_realistic'/p) for p in ['trajectory.csv', 'fruit_trajectory.csv']})
    if reference:
        result['video_mean_reduction_percent'] = 100*(1-stats['candidate']['video']['mean_ms']/stats['reference']['video']['mean_ms'])
        result['offscreen_mean_reduction_percent'] = 100*(1-stats['candidate']['offscreen']['mean_ms']/stats['reference']['offscreen']['mean_ms'])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    for variant, modes in stats.items():
        print(variant, 'video mean %.2f ms; off-image mean %.2f ms' % (modes['video']['mean_ms'], modes['offscreen']['mean_ms']))
    print('Matching output files:' if reference else 'Saved output files:', len(golden), 'Results:', output)


if __name__ == '__main__':
    main()
