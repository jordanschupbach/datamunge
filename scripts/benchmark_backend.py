#!/usr/bin/env python3
"""Build/run reproducible Google Benchmark suites; compare JSON without CPM tools."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

from tensorflow_fixtures import CASES as TF_CASES

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build' / 'benchmarks'
RESULTS = ROOT / 'benchmarks' / 'results'


def call(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, **kwargs)


def revision():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT)
    sources = {}
    for directory in ('benchmarks/backend',):
        for p in sorted((ROOT / directory).glob('*')):
            if p.is_file():
                sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    catalog = ROOT / 'benchmarks/tensorflow_cases.def'
    if catalog.is_file():
        sources[str(catalog.relative_to(ROOT))] = hashlib.sha256(catalog.read_bytes()).hexdigest()
    # Include new, not-yet-tracked private kernel headers as well as tracked core
    # sources; git diff alone cannot identify their contents before a commit.
    core = hashlib.sha256()
    for directory in ('src/datamunge', 'include/datamunge'):
        for p in sorted((ROOT / directory).rglob('*')):
            if p.is_file() and p.suffix in ('.cpp', '.hpp', '.h', '.c'):
                core.update(str(p.relative_to(ROOT)).encode() + b'\0' + p.read_bytes() + b'\0')
    return {'revision': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=normal')),
            'working_diff_sha256': hashlib.sha256(diff).hexdigest(), 'benchmark_sources_sha256': sources,
            'core_sources_sha256': core.hexdigest()}


def optimized_release_command(command):
    args = command.get('arguments') or shlex.split(command['command'], posix=os.name != 'nt')
    opts = [arg for arg in args if re.fullmatch(r'(?:-O(?:[0-9]+|g|s|z|fast)|/O[12dx])', arg)]
    ndebug = False
    for i, arg in enumerate(args):
        if re.fullmatch(r'(?:-D|/D)NDEBUG(?:=.*)?', arg):
            ndebug = True
        elif arg in ('-UNDEBUG', '/UNDEBUG'):
            ndebug = False
        elif arg in ('-D', '/D', '-U', '/U') and i + 1 < len(args) and args[i + 1].split('=')[0] == 'NDEBUG':
            ndebug = arg in ('-D', '/D')
    return bool(opts and opts[-1] in ('-O2', '-O3', '/O2') and ndebug)


def release_configuration(configuration):
    types = [value for value in configuration.get('CMAKE_CONFIGURATION_TYPES', '').split(';') if value]
    return types == ['Release'] if types else configuration.get('CMAKE_BUILD_TYPE') == 'Release'


def release_manifest(build, suite='backend_bench'):
    commands = json.loads((build / 'compile_commands.json').read_text())
    core = [c for c in commands if '/src/datamunge/' in c['file']]
    directory = {'backend_bench': 'backend', 'nn_bench': 'nn', 'linalg_bench': 'cpp'}[suite]
    drivers = [c for c in commands if '/benchmarks/' + directory + '/' in c['file']]
    if not core or not drivers:
        raise RuntimeError('Missing core or benchmark compiler commands')
    for c in core + drivers:
        if not optimized_release_command(c):
            raise RuntimeError(f"Core/driver is not an optimized Release build: {c['file']}")
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line or ':' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.split(':', 1)[0]
        if key in ('CMAKE_BUILD_TYPE', 'CMAKE_CONFIGURATION_TYPES', 'CMAKE_CXX_COMPILER', 'CMAKE_CXX_FLAGS',
                   'CMAKE_CXX_FLAGS_RELEASE', 'DATAMUNGE_USE_BLAS', 'DATAMUNGE_USE_LAPACK',
                   'DATAMUNGE_ENABLE_ARROW'):
            cache[key] = value
    if not release_configuration(cache):
        raise RuntimeError('Benchmarks must be configured as Release (including multi-config generators)')
    return {'configuration': cache, 'core_translation_units': len(core), 'driver_translation_units': len(drivers),
            'effective_optimization_verified': True, 'ndebug_verified': True, 'core_command_example': core[0],
            'driver_command_example': drivers[0]}


def release_runtime_tags(context):
    if context.get('library_build_type') != 'release':
        raise RuntimeError('Google Benchmark runtime is not Release')
    tag = context.get('build')
    if tag and tag != 'Release' and not tag.endswith('/Release'):
        raise RuntimeError('Benchmark driver runtime is not Release')
    return {'google_benchmark': 'release', 'driver': tag}


def release_runtime(context, build, core_provenance=None, require_core=False):
    proof = release_runtime_tags(context)
    library = context.get('core_library_path')
    if not library:
        if core_provenance or (require_core and os.name == 'posix'):
            raise RuntimeError('Missing loaded core library identity')
        return proof
    library = Path(library).resolve()
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    proof.update(core_library_path=str(library), core_sha256=digest)
    if core_provenance:
        reference = json.loads(Path(core_provenance).read_text())
        metadata = reference.get('backend_benchmark_metadata', {})
        manifest = metadata.get('release', {})
        if not release_configuration(manifest.get('configuration', {})) or not all(
                optimized_release_command(manifest.get(key, {'command': ''}))
                for key in ('core_command_example', 'driver_command_example')):
            raise RuntimeError('Core provenance does not attest an optimized Release build')
        release_runtime_tags(reference.get('context', {}))
        if digest != metadata.get('loaded_core_sha256'):
            raise RuntimeError('Loaded core hash does not match its Release provenance')
        proof['core_release_provenance'] = str(Path(core_provenance).resolve())
    else:
        try:
            library.relative_to(Path(build).resolve())
        except ValueError:
            raise RuntimeError('Loaded core is outside the verified Release build; supply --core-provenance')
        proof['core_release_provenance'] = 'verified build directory'
    return proof


# One small case per operation/model/mode, including both tensor layouts and dtypes.
SMOKE = (r'BM_TensorLinalg/.*/n:16/batch:1/|BM_TensorMatmul/m:16/k:64/n:32/batch:1/'
         r'|BM_TensorOps/.*/elements:1024/|BM_LinalgTensor.*/1024/'
         r'|BM_AutogradChain/.*/elements:32/depth:16/|BM_AutogradFanout/1/'
         r'|BM_AutogradLinalg/.*/16/|BM_AutodiffTape/.*/16/'
         r'|BM_GeneralOptimizer/.*/dimension:2/problem:0/'
         r'|BM_GeneralOptimizer/(DifferentialEvolution|PSO|PSORing)/2/0/42/'
         r'|BM_Constrained|BM_MultiObjectiveNSGA2/42/|BM_Network/.*/1/'
         r'|BM_TensorOptimizer/.*/elements:4096/tensors:1/'
         r'|BM_AutodiffJacobian/8/|BM_AutodiffForward/.*/4/|BM_LinalgTensorMatmul/16/|BM_TensorRectangular'
         r'|BM_TensorParallelScaling|BM_ParallelSubmit|BM_TensorThreadScaling|BM_GradientUtilities/.*/4096/'
         r'|BM_AttentionSmall/tokens:16/|BM_NeuralPrimitive/.*/batch:32/'
         r'|BM_TensorDetEig/.*/16/|BM_TensorEinsum/1/'
         r'|BM_TensorBroadcast|BM_TensorMax/1024/|BM_LinalgTensorExtra/.*/1024/'
         r'|BM_SquarePrimitiveVjp/rows:512/'
         r'|BM_ContainerParameters/.*/16/|BM_ContainerOptimizer/.*/16/|BM_ContainerBatchNorm/32/1/|BM_ContainerFFT/1/|BM_ContainerGraph/16/'
         r'|BM_TensorSolveRhs/n:16/|BM_GeneralEigenvectors/16/'
         r'|BM_LinalgTensorTransposeLayout/rows:256/cols:256/|BM_TensorMatmulPartialBatch/width:16/|BM_TensorSolveShared/n:16/|BM_TensorMatmulShared/batch:16/n:32/'
         r'|BM_LayerComponent/.*/batch:1/|BM_OptimizerTimeToTarget/|BM_AutodiffBranch/16/')


# Keep optional TensorFlow imports out of native CI; only the shared data catalog is read.
SMOKE += '|' + '|'.join(re.escape(f'BM_TensorFlowMatched/{op}/rows:{sizes[0]}/')
                         for op, sizes in TF_CASES.items())


def audit(build):
    listing = subprocess.check_output([str(build / 'backend_bench'), '--benchmark_list_tests=true'], cwd=ROOT, text=True)
    names = listing.splitlines()
    manifest = json.loads((ROOT / 'benchmarks/coverage.json').read_text())
    missing = []
    for area, patterns in manifest['required_patterns'].items():
        for pattern in patterns:
            if not any(re.search(pattern, name) for name in names):
                missing.append(f'{area}: {pattern}')
    # Detect optimizer additions in the actual public enum, instead of trusting a hard-coded count.
    header = (ROOT / 'include/datamunge/nn/optim.hpp').read_text()
    enum = re.search(r'enum class OptimizerKind\s*\{(.*?)\};', header, re.S).group(1)
    enum = re.sub(r'//[^\n]*', '', enum)
    kinds = [v.strip() for v in enum.split(',') if v.strip()]
    for i, kind in enumerate(kinds):
        for phase in ('FirstStep', 'Warmed', 'DoubleFirstStep', 'DoubleWarmed'):
            for vector in (0, 1):
                pattern = rf'BM_TensorOptimizer/{phase}/.*kind:{i}/vector:{vector}/'
                if not any(re.search(pattern, name) for name in names):
                    missing.append(f'tensor optimizer: {kind}, {phase}, vector={vector}')
    if missing:
        raise RuntimeError('Missing benchmark coverage:\n' + '\n'.join(missing))
    print(f'Coverage audit passed: {len(names)} registered cases; {len(manifest["required_patterns"])} areas; all {len(kinds)} tensor optimizers in both dtypes, phases and shapes')
    return names


def run(args, extra):
    build = Path(args.build_dir).resolve()
    if not args.no_build:
        call(['cmake', '-S', 'benchmarks', '-B', str(build), '-DCMAKE_BUILD_TYPE=Release', *args.cmake_arg])
        call(['cmake', '--build', str(build), '--config', 'Release', '--target', args.suite, '-j', str(args.jobs)])
    manifest = release_manifest(build, args.suite)
    if args.suite == 'backend_bench':
        audit(build)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path(args.output).resolve() if args.output else RESULTS / f'{args.suite}_{stamp}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(build / args.suite), '--benchmark_out=' + str(output), '--benchmark_out_format=json']
    if args.smoke:
        if args.suite != 'backend_bench':
            raise RuntimeError('--smoke is defined for backend_bench; filter legacy suites explicitly')
        cmd += ['--benchmark_filter=' + SMOKE, '--benchmark_min_time=1x']
    cmd += extra
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
        env.setdefault(name, '1')
    metadata = {**revision(), 'platform': platform.platform(), 'machine': platform.machine(),
                'command': cmd, 'release': manifest, 'label': args.label,
                'executable_sha256': hashlib.sha256((build / args.suite).read_bytes()).hexdigest(),
                'thread_environment': {k: env.get(k) for k in ('DATAMUNGE_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')}}
    call(cmd, env=env)
    data = json.loads(output.read_text())
    metadata['release_runtime'] = release_runtime(data.get('context', {}), build,
                                                 getattr(args, 'core_provenance', None), args.suite == 'backend_bench')
    library = data.get('context', {}).get('core_library_path')
    if library and Path(library).is_file():
        metadata['loaded_core_sha256'] = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    data['backend_benchmark_metadata'] = metadata
    output.write_text(json.dumps(data, indent=2) + '\n')
    rows = data.get('benchmarks', [])
    errors = [r for r in rows if r.get('error_occurred')]
    # An unavailable optional LAPACK mode is an explicit skip, not a failed fixture.
    failures = [r for r in errors if r.get('error_message') != 'LAPACK unavailable']
    if not rows or failures:
        raise RuntimeError(f'Benchmark run failed: {failures or "no cases selected"}')
    record_result(output)
    print(f'Saved {len(rows)} measurements ({len(errors)} optional skips): {output}')


def compare(args):
    def load(path):
        p = Path(path)
        if not p.exists():
            p = RESULTS / p
        return json.loads(p.read_text())
    base, new = load(args.base), load(args.new)
    if bool(base.get('backend_benchmark_metadata')) != bool(new.get('backend_benchmark_metadata')):
        raise RuntimeError('Cannot compare a metadata-backed run with a legacy run')
    if base.get('backend_benchmark_metadata', {}).get('release', {}).get('configuration') != new.get('backend_benchmark_metadata', {}).get('release', {}).get('configuration'):
        raise RuntimeError('Incompatible Release build configurations')
    if not base.get('backend_benchmark_metadata'):
        print('Warning: legacy results have no verified build/thread metadata', file=sys.stderr)
    for key in ('platform', 'machine', 'thread_environment'):
        a = base.get('backend_benchmark_metadata', {}).get(key)
        b = new.get('backend_benchmark_metadata', {}).get(key)
        if a != b:
            raise RuntimeError(f'Incompatible benchmark metadata: {key}')
    for key in ('build', 'gemm_backend', 'lapack_available', 'nn_threads', 'num_cpus', 'mhz_per_cpu', 'caches'):
        if base.get('context', {}).get(key) != new.get('context', {}).get(key):
            raise RuntimeError(f'Incompatible benchmark context: {key}')
    for data in (base, new):
        failed = [r for r in data['benchmarks'] if r.get('error_occurred') and r.get('error_message') != 'LAPACK unavailable']
        if failed:
            raise RuntimeError('Cannot compare results containing benchmark errors')
    def rows(data):
        grouped = {}
        for r in data['benchmarks']:
            if r.get('error_occurred') or r.get('run_type') == 'aggregate':
                continue
            if r['time_unit'] not in ('ns', 'us', 'ms', 's'):
                raise RuntimeError('Unknown time unit')
            scale = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1}[r['time_unit']]
            grouped.setdefault(r['name'], []).append(r['real_time'] * scale)
        import statistics
        return {name: statistics.median(values) for name, values in grouped.items()}
    a, b = rows(base), rows(new)
    common = sorted(a.keys() & b.keys())
    if not common:
        raise RuntimeError('No comparable cases')
    print('case\tbaseline_ms\tnew_ms\tchange_percent')
    for name in common:
        if a[name] <= 0 or not math.isfinite(a[name]) or not math.isfinite(b[name]):
            raise RuntimeError('Invalid timing')
        print(f'{name}\t{a[name]*1e3:.6g}\t{b[name]*1e3:.6g}\t{100*(b[name]/a[name]-1):+.2f}')
    print(f'{len(common)} comparable cases; {len(a.keys()-b.keys())} removed, {len(b.keys()-a.keys())} added. Negative is faster.')


# Compact summaries are versionable; raw content-addressed files are retained
# separately so summaries never depend on a mutable output filename.
HISTORY = ROOT / 'benchmarks' / 'history'


def record_result(path, directory=HISTORY, label=None):
    import statistics
    import tempfile
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    metadata = data.get('backend_benchmark_metadata')
    if not metadata:
        raise RuntimeError('History requires a result with build/revision metadata')
    grouped = {}
    skipped = 0
    for row in data.get('benchmarks', []):
        if row.get('error_occurred'):
            if row.get('error_message') != 'LAPACK unavailable':
                raise RuntimeError('Cannot record failed benchmark results')
            skipped += 1
            continue
        if row.get('run_type') == 'aggregate':
            continue
        scale = {'ns': 1, 'us': 1000, 'ms': 1000000, 's': 1000000000}.get(row.get('time_unit'))
        if scale is None or not math.isfinite(row['real_time']) or row['real_time'] <= 0:
            raise RuntimeError('Cannot record invalid benchmark timings')
        normalized = row['real_time'] * scale
        if not math.isfinite(normalized):
            raise RuntimeError('Cannot record invalid normalized timing')
        grouped.setdefault(row.get('run_name', row['name']), []).append((row, normalized))
    if not grouped:
        raise RuntimeError('Cannot record an empty benchmark run')
    cases = {}
    for name, rows in sorted(grouped.items()):
        times = [time for _, time in rows]
        counters = {}
        for key in rows[0][0]:
            if key.endswith('_tensor_bytes') or key in ('objective_calls/solve', 'gradient_calls/solve',
                                                       'hessian_calls/solve', 'final_objective', 'target_met',
                                                       'items_per_second', 'bytes_per_second', 'FLOP/s'):
                values = [row[key] for row, _ in rows if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
                if values:
                    counters[key] = statistics.median(values)
        cases[name] = {'median_ns': statistics.median(times), 'min_ns': min(times), 'max_ns': max(times),
                       'cv_percent': statistics.stdev(times) / statistics.mean(times) * 100 if len(times) > 1 else None,
                       'repetitions': len(times), 'counters': counters}
    command = metadata.get('command', [])
    duration = next((value.split('=', 1)[1] for value in command if value.startswith('--benchmark_min_time=')), '')
    seconds = re.fullmatch(r'(\d+(?:\.\d+)?)(?:s)?', duration)
    repeated = min(case['repetitions'] for case in cases.values()) >= 3
    kind = 'smoke' if duration.endswith('x') else 'calibrated' if repeated and seconds and float(seconds[1]) >= 0.1 else 'exploratory'
    context = data.get('context', {})
    configuration = {key: metadata.get(key) for key in ('platform', 'machine', 'thread_environment')}
    configuration['release'] = metadata.get('release', {}).get('configuration')
    configuration['context'] = {key: context.get(key) for key in ('build', 'gemm_backend', 'lapack_available',
                                                               'nn_threads', 'num_cpus', 'mhz_per_cpu', 'caches')}
    profile = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    run_id = hashlib.sha256(raw).hexdigest()
    record = {'schema_version': 1, 'id': run_id, 'date': context.get('date'),
              'label': label or metadata.get('label'), 'kind': kind, 'profile': profile,
              'configuration': configuration, 'provenance': metadata, 'optional_skips': skipped, 'cases': cases}
    directory = Path(directory)
    for folder, content in [('runs', raw), ('summaries', (json.dumps(record, indent=2) + '\n').encode())]:
        target = directory / folder / (run_id + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if folder == 'runs' and target.read_bytes() != raw:
                raise RuntimeError(f'History archive integrity check failed: {target}')
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f'Recorded {kind} history: {run_id} ({len(cases)} cases; profile {profile[:12]})')
    return run_id


def history_date(record):
    value = record.get('date') or ''
    if not isinstance(value, str):
        return str(value)
    try:
        parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is not None:
            return parsed.astimezone(datetime.timezone.utc).isoformat()
    except ValueError:
        pass
    return value


def history(args):
    pattern = re.compile(args.case)
    print('date\trun_id\tprofile\tlabel\tkind\tcase\tmedian_ms\tcv_percent\trepetitions')
    records = [json.loads(path.read_text()) for path in (Path(args.directory) / 'summaries').glob('*.json')]
    for record in sorted(records, key=lambda item: (history_date(item), item['id'])):
        if not args.include_smoke and record['kind'] != 'calibrated':
            continue
        if args.profile and not record['profile'].startswith(args.profile):
            continue
        for name, case in record['cases'].items():
            if pattern.search(name):
                cv = case['cv_percent']
                values = (history_date(record), record['id'], record['profile'], record.get('label') or '',
                          record['kind'], name, f"{case['median_ns'] / 1e6:.6g}",
                          '' if cv is None else f'{cv:.3g}', str(case['repetitions']))
                print('\t'.join(str(value).replace('\t', ' ').replace('\n', ' ') for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--suite', choices=('backend_bench', 'nn_bench', 'linalg_bench'), default='backend_bench')
    r.add_argument('--build-dir', default=str(BUILD))
    r.add_argument('--jobs', type=int, default=4)
    r.add_argument('--cmake-arg', action='append', default=[])
    r.add_argument('--no-build', action='store_true')
    r.add_argument('--core-provenance', type=Path, help='Release-backed result attesting an overridden/frozen core hash')
    r.add_argument('--smoke', action='store_true')
    r.add_argument('--output')
    r.add_argument('--label', help='Description stored with the persistent run')
    a = sub.add_parser('audit')
    a.add_argument('--build-dir', default=str(BUILD))
    c = sub.add_parser('compare')
    c.add_argument('base'); c.add_argument('new')
    record = sub.add_parser('record', help='Archive a previously collected result')
    record.add_argument('result')
    record.add_argument('--directory', default=str(HISTORY))
    record.add_argument('--label')
    hist = sub.add_parser('history', help='Print long-term median/dispersion history as TSV')
    hist.add_argument('--directory', default=str(HISTORY))
    hist.add_argument('--case', default='.*', help='Case-name regular expression')
    hist.add_argument('--profile', help='Restrict to a compatible machine/build/thread profile hash prefix')
    hist.add_argument('--include-smoke', action='store_true', help='Also show smoke/exploratory observations')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    if args.action == 'run':
        run(args, extra)
    elif args.action == 'audit':
        if extra:
            parser.error('Unexpected audit arguments')
        audit(Path(args.build_dir).resolve())
    elif args.action in ('record', 'history'):
        if extra:
            parser.error('Unexpected history arguments')
        if args.action == 'record':
            record_result(args.result, args.directory, args.label)
        else:
            history(args)
    else:
        if extra:
            parser.error('Unexpected comparison arguments')
        compare(args)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError, json.JSONDecodeError, re.error) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
