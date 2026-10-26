#!/usr/bin/env python3
"""Matched CPU comparison: native C++ vs TensorFlow eager and traced graphs.

Run with the isolated TensorFlow Python; no framework dependency in the core.
Native measurements are produced by benchmark_backend.py and may be reused.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

import benchmark_backend as backend
from performance_report import update_report

from tensorflow_fixtures import CATALOG, CASES, OPERATIONS, fixture


def selected_cases(pattern='.*'):
    regex = re.compile(pattern)
    names = {f'BM_TensorFlowMatched/{op}/rows:{rows}/f64:{f64}/real_time'
             for op in OPERATIONS for rows in CASES[op]
             for f64 in (0, 1) if regex.search(f'{op}/rows:{rows}/f64:{f64}/real_time')}
    if not names:
        raise RuntimeError('Case filter matched no fixtures')
    return names


def measure(run, seconds, repetitions):
    for _ in range(10):
        run()
    count = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(count):
            run()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if elapsed >= seconds:
            break
        count = max(count + 1, min(count * 10, int(count * seconds / max(elapsed, 1e-9) * 1.1)))
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for _ in range(count):
            run()
        samples.append((time.perf_counter_ns() - start) / count)
    # CPU synchronous execution is explicitly enabled; avoid host copies in timing.
    return count, samples


def native_case(args, suffix, native):
    """Adjacent framework measurements limit drift during a long suite run."""
    executable = args.build_dir.resolve() / 'backend_bench'
    manifest = native['backend_benchmark_metadata']['release'] if native else backend.release_manifest(args.build_dir.resolve())
    with tempfile.TemporaryDirectory(prefix='tensorflow-native-') as directory:
        output = Path(directory) / 'native.json'
        command = [str(executable), '--benchmark_filter=^BM_TensorFlowMatched/' + suffix + '$',
                   f'--benchmark_min_time={args.min_time}s', f'--benchmark_repetitions={args.repetitions}',
                   '--benchmark_out=' + str(output), '--benchmark_out_format=json']
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        observation = json.loads(output.read_text())
    runtime = backend.release_runtime(observation['context'], args.build_dir,
                                      getattr(args, 'core_provenance', None), require_core=True)
    rows = [row for row in observation['benchmarks'] if row.get('run_type') != 'aggregate']
    if len(rows) != args.repetitions or any(row.get('error_occurred') or row.get('reference_checked') != 1 for row in rows):
        raise RuntimeError(f'Native paired fixture failed: {suffix}')
    if not native:
        metadata = {**backend.revision(), 'platform': platform.platform(), 'machine': platform.machine(),
                    'label': args.label + '-native-paired', 'release': manifest, 'release_runtime': runtime,
                    'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                    'thread_environment': {key: os.environ.get(key) for key in
                        ('DATAMUNGE_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')},
                    'command': [str(executable), f'--benchmark_min_time={args.min_time}s',
                                f'--benchmark_repetitions={args.repetitions}'],
                    'timing_order': 'adjacent to TensorFlow; native first/last alternates per fixture',
                    'paired_case_filters': [],
                    'paired_case_load_averages': [],
                    'host_load_average_start': os.getloadavg() if hasattr(os, 'getloadavg') else None}
        library = observation['context'].get('core_library_path')
        if library and Path(library).is_file():
            metadata['loaded_core_sha256'] = hashlib.sha256(Path(library).read_bytes()).hexdigest()
        native.update({'context': observation['context'], 'backend_benchmark_metadata': metadata, 'benchmarks': []})
    elif native['context'] != observation['context']:
        # Date necessarily changes between subprocesses; all hardware/build fields must agree.
        left, right = dict(native['context']), dict(observation['context'])
        for key in ('date', 'load_avg'):
            left.pop(key, None); right.pop(key, None)
        if left != right:
            raise RuntimeError('Native build/hardware context changed during paired run')
    native['backend_benchmark_metadata']['paired_case_filters'].append(command[1])
    if runtime != native['backend_benchmark_metadata']['release_runtime']:
        raise RuntimeError('Loaded core identity changed during paired run')
    native['backend_benchmark_metadata']['paired_case_load_averages'].append(observation['context'].get('load_avg'))
    native['benchmarks'].extend(observation['benchmarks'])


def compare(native, tensorflow, expected_names=None):
    groups = {}
    for row in native['benchmarks']:
        if row.get('error_occurred'):
            raise RuntimeError('Native reference/timing failure')
        if row.get('run_type') == 'aggregate':
            continue
        name = row.get('run_name', row['name'])
        if not name.startswith('BM_TensorFlowMatched/'):
            continue
        scale = {'ns': 1, 'us': 1000, 'ms': 1000000, 's': 1000000000}[row['time_unit']]
        if not math.isfinite(row['real_time']) or row['real_time'] <= 0:
            raise RuntimeError('Invalid native timing')
        groups.setdefault(name, []).append(row['real_time'] * scale)
    tf_groups = {}
    for row in tensorflow['benchmarks']:
        if row.get('error_occurred') or not math.isfinite(row['real_time']) or row['real_time'] <= 0:
            raise RuntimeError('Invalid TensorFlow timing')
        tf_groups.setdefault(row['name'], []).append(row['real_time'])
    if expected_names is not None and set(groups) != set(expected_names):
        raise RuntimeError('Native fixtures do not match the requested case selection')
    comparisons = []
    for name, native_times in groups.items():
        values = {}
        for mode in ('eager', 'graph'):
            tf_name = name.replace('BM_TensorFlowMatched/', f'TensorFlow/{mode}/', 1)
            if tf_name not in tf_groups:
                raise RuntimeError(f'Missing TensorFlow case: {tf_name}')
            values[mode + '_ns'] = statistics.median(tf_groups[tf_name])
            samples = tf_groups[tf_name]
            values[mode + '_cv_percent'] = statistics.stdev(samples) / statistics.mean(samples) * 100 if len(samples) > 1 else 0
        native_ns = statistics.median(native_times)
        comparisons.append({'case': name, 'native_ns': native_ns, **values,
                            'native_cv_percent': statistics.stdev(native_times) / statistics.mean(native_times) * 100 if len(native_times) > 1 else 0,
                            'native_over_eager': native_ns / values['eager_ns'],
                            'native_over_graph': native_ns / values['graph_ns']})
    expected_count = len(selected_cases()) if expected_names is None else len(expected_names)
    if len(comparisons) != expected_count:
        raise RuntimeError(f'Expected {expected_count} matched cases, got {len(comparisons)}')
    return sorted(comparisons, key=lambda row: row['native_over_eager'], reverse=True)


def validate(args, np, tf):
    selection = selected_cases(args.case)
    output = args.output.with_suffix('.native.json')
    subprocess.run([sys.executable, str(backend.ROOT / 'scripts/benchmark_backend.py'),
                    'run', '--no-build', '--build-dir', str(args.build_dir), '--output', str(output),
                    '--label', args.label + '-validation', '--',
                    '--benchmark_filter=^(' + '|'.join(re.escape(n) for n in sorted(selection)) + ')$',
                    '--benchmark_min_time=1x'], check=True, stdout=subprocess.DEVNULL)
    native = json.loads(output.read_text())
    successful = {r['name'] for r in native['benchmarks'] if r.get('reference_checked') == 1 and not r.get('error_occurred')}
    if successful != selection:
        raise RuntimeError('Native validation did not pass every requested fixture')
    for name in sorted(selection):
        op, scale, precision, _ = name.removeprefix('BM_TensorFlowMatched/').split('/')
        rows, f64 = int(scale.split(':')[1]), int(precision.split(':')[1])
        with tf.device('/CPU:0'):
            inputs, operation, reference, tolerance = fixture(op, rows, f64, np, tf)
            for mode in ('eager', 'graph'):
                fn = operation if mode == 'eager' else tf.function(operation, autograph=False, jit_compile=False)
                actual = fn(*inputs).numpy()
                if actual.shape != np.shape(reference): raise RuntimeError(f'Wrong shape for {name}')
                np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)
        print('Validated ' + name, flush=True)
    args.output.write_text(json.dumps({'kind': 'validation-only', 'cases': sorted(selection),
                                      'tensorflow': tf.__version__, 'native_result': str(output)}, indent=2) + '\n')
    print(f'Validated {len(selection)} native/eager/graph fixture triples')


def run(args):
    selection = selected_cases(args.case)
    if not args.reuse_native:
        backend.release_manifest(args.build_dir.resolve())  # Fail before import/warmup/timing.
    for key in ('DATAMUNGE_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf

    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.set_synchronous_execution(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        return validate(args, np, tf)
    args.native.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_native and not args.paired_native:
        subprocess.run([sys.executable, str(backend.ROOT / 'scripts/benchmark_backend.py'),
                        'run', '--no-build', '--build-dir', str(args.build_dir),
                        '--label', args.label + '-native', '--output', str(args.native),
                        *(['--core-provenance', str(args.core_provenance)] if args.core_provenance else []), '--',
                        '--benchmark_filter=^(' + '|'.join(re.escape(name) for name in sorted(selection)) + ')$',
                        f'--benchmark_min_time={args.min_time}s',
                        f'--benchmark_repetitions={args.repetitions}'], check=True)
    native = {} if args.paired_native else json.loads(args.native.read_text())
    if args.reuse_native:
        metadata = native.get('backend_benchmark_metadata', {})
        manifest = metadata.get('release', {})
        if not backend.release_configuration(manifest.get('configuration', {})) or not all(
                backend.optimized_release_command(manifest.get(key, {'command': ''}))
                for key in ('core_command_example', 'driver_command_example')):
            raise RuntimeError('Reused native results do not attest an optimized Release build')
        backend.release_runtime_tags(native.get('context', {}))
    env = {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                              'OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')}
    data = {'context': {'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'build': 'TensorFlow-' + tf.__version__, 'gemm_backend': 'TensorFlow packaged CPU kernels',
                        'nn_threads': '1', 'tensorflow_build': tf.sysconfig.get_build_info()},
            'backend_benchmark_metadata': {
                **backend.revision(), 'platform': platform.platform(), 'machine': platform.machine(),
                'label': args.label + '-tensorflow', 'thread_environment': env,
                'release': {'configuration': {'tensorflow': tf.__version__, 'numpy': np.__version__}},
                'command': [*sys.argv, f'--benchmark_min_time={args.min_time}s'],
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'case_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
                'fixture_factory_sha256': hashlib.sha256(Path(__file__).with_name('tensorflow_fixtures.py').read_bytes()).hexdigest(),
                'native_result_sha256': None if args.paired_native else hashlib.sha256(args.native.read_bytes()).hexdigest(),
                'device': '/CPU:0', 'synchronous': True,
                'case_selection': sorted(selection),
                'host_load_average_start': os.getloadavg() if hasattr(os, 'getloadavg') else None,
                'timing_scope': 'warm synchronous call; output allocation included; host copy excluded; graph trace excluded',
                'python': sys.version, 'native_result': str(args.native)},
            'benchmarks': []}
    case_index = 0
    for op in OPERATIONS:
        for rows in CASES[op]:
            for f64 in (0, 1):
                suffix = f'{op}/rows:{rows}/f64:{f64}/real_time'
                if 'BM_TensorFlowMatched/' + suffix not in selection:
                    continue
                if args.paired_native and case_index % 2 == 0:
                    native_case(args, suffix, native)
                with tf.device('/CPU:0'):
                    inputs, operation, reference, tolerance = fixture(op, rows, f64, np, tf)
                    for mode in ('eager', 'graph'):
                        fn = operation if mode == 'eager' else tf.function(operation, autograph=False, jit_compile=False)
                        actual = fn(*inputs).numpy()
                        if actual.shape != np.shape(reference):
                            raise RuntimeError(f'Wrong output shape: {op}')
                        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)
                        count, samples = measure(lambda: fn(*inputs), args.min_time, args.repetitions)
                        name = f'TensorFlow/{mode}/{op}/rows:{rows}/f64:{f64}/real_time'
                        for repetition, sample in enumerate(samples):
                            data['benchmarks'].append({'name': name, 'run_name': name, 'run_type': 'iteration',
                                                       'repetition_index': repetition, 'iterations': count,
                                                       'real_time': sample, 'time_unit': 'ns', 'reference_checked': 1,
                                                       'reference_tolerance': tolerance,
                                                       'max_absolute_reference_error': float(np.max(np.abs(actual - reference)))})
                        print(f'{name}: {statistics.median(samples) / 1000:.3f} us', flush=True)
                if args.paired_native and case_index % 2 == 1:
                    native_case(args, suffix, native)
                case_index += 1
    if args.paired_native:
        native['backend_benchmark_metadata']['host_load_average_end'] = os.getloadavg() if hasattr(os, 'getloadavg') else None
        args.native.write_text(json.dumps(native, indent=2) + '\n')
        backend.record_result(args.native)
        data['backend_benchmark_metadata']['native_result_sha256'] = hashlib.sha256(args.native.read_bytes()).hexdigest()
    data['backend_benchmark_metadata']['host_load_average_end'] = os.getloadavg() if hasattr(os, 'getloadavg') else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    backend.record_result(args.output)
    comparisons = compare(native, data, selection)
    args.output.with_suffix('.comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    with args.output.with_suffix('.comparison.tsv').open('w') as stream:
        stream.write('case\tnative_us\teager_us\tgraph_us\tnative_over_eager\tnative_over_graph\tnative_cv_percent\teager_cv_percent\tgraph_cv_percent\n')
        for row in comparisons:
            stream.write(f"{row['case']}\t{row['native_ns']/1000:.6g}\t{row['eager_ns']/1000:.6g}\t"
                         f"{row['graph_ns']/1000:.6g}\t{row['native_over_eager']:.6g}\t{row['native_over_graph']:.6g}\t"
                         f"{row['native_cv_percent']:.3g}\t{row['eager_cv_percent']:.3g}\t{row['graph_cv_percent']:.3g}\n")
    print(f'Saved {len(comparisons)} matched comparisons: {args.output.with_suffix(".comparison.tsv")}')
    if args.output.resolve().parent == (backend.RESULTS / 'tensorflow').resolve():
        update_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=backend.RESULTS / 'tensorflow/native.json')
    parser.add_argument('--output', type=Path, default=backend.RESULTS / 'tensorflow/tensorflow.json')
    parser.add_argument('--build-dir', type=Path, default=backend.BUILD)
    parser.add_argument('--core-provenance', type=Path, help='Release-backed result attesting an overridden/frozen core hash')
    parser.add_argument('--validate-only', action='store_true', help='Check all outputs without collecting calibrated timings')
    parser.add_argument('--reuse-native', action='store_true')
    parser.add_argument('--paired-native', action='store_true', help='Measure native adjacent to each TensorFlow fixture, alternating order')
    parser.add_argument('--case', default='.*', help='Regex over Operation/rows:N/f64:N/real_time; default runs the complete shared catalog')
    parser.add_argument('--label', default='tensorflow-comparison')
    parser.add_argument('--min-time', type=float, default=0.1)
    parser.add_argument('--repetitions', type=int, default=5)
    options = parser.parse_args()
    if not math.isfinite(options.min_time) or options.min_time < 0.1 or options.repetitions < 3:
        parser.error('Comparisons require at least 0.1 seconds and three repetitions')
    if options.paired_native and options.reuse_native:
        parser.error('--paired-native cannot reuse a previous native run')
    run(options)
