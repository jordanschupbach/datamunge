"""Contract tests for benchmark comparison and Release verification (no compiler needed)."""
import contextlib
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location('benchmark_backend', Path(__file__).resolve().parents[1] / 'scripts/benchmark_backend.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_effective_release_flags_and_ndebug(self):
        for command, expected in [
            ('c++ -O3 -DNDEBUG -g -c a.cpp', True),
            ('c++ -O0 -O3 -D NDEBUG -c a.cpp', True),
            ('c++ -O3 -DNDEBUG -O0 -c a.cpp', False),
            ('c++ -O3 -DNDEBUG -UNDEBUG -c a.cpp', False),
            ('c++ -O3 -DNDEBUG -U NDEBUG -c a.cpp', False),
            ('c++ -O3 -DNDEBUG_OTHER -c a.cpp', False),
            ('cl /O2 /DNDEBUG /c a.cpp', True),
            ('cl /O2 /Ob2 /DNDEBUG /c a.cpp', True),
            ('cl /O2 /DNDEBUG /Od /c a.cpp', False),
        ]:
            with self.subTest(command=command):
                self.assertEqual(runner.optimized_release_command({'command': command}), expected)

    def test_optimized_flags_alone_do_not_make_debug_configuration_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'compile_commands.json').write_text(json.dumps([
                {'file': '/repo/src/datamunge/a.cpp', 'command': 'c++ -O3 -DNDEBUG -c a.cpp'},
                {'file': '/repo/benchmarks/backend/main.cpp', 'command': 'c++ -O3 -DNDEBUG -c main.cpp'}]))
            cache = root / 'CMakeCache.txt'
            cache.write_text('CMAKE_BUILD_TYPE:STRING=Debug\n')
            with self.assertRaisesRegex(RuntimeError, 'configured as Release'):
                runner.release_manifest(root)
            cache.write_text('CMAKE_BUILD_TYPE:STRING=Release\n')
            self.assertTrue(runner.release_manifest(root)['effective_optimization_verified'])
            self.assertFalse(runner.release_configuration({'CMAKE_CONFIGURATION_TYPES': 'Debug;Release'}))
            self.assertTrue(runner.release_configuration({'CMAKE_CONFIGURATION_TYPES': 'Release'}))

    def test_runtime_and_external_core_release_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build = root / 'build'; build.mkdir()
            library = root / 'external-core.dylib'; library.write_bytes(b'compiled release core')
            context = {'build': 'Clang/Release', 'library_build_type': 'release', 'core_library_path': str(library)}
            with self.assertRaisesRegex(RuntimeError, 'outside the verified Release build'):
                runner.release_runtime(context, build)
            provenance = root / 'reference.json'
            command = {'command': 'clang++ -O3 -DNDEBUG -c core.cpp'}
            provenance.write_text(json.dumps({'context': context, 'backend_benchmark_metadata': {
                'loaded_core_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'release': {'configuration': {'CMAKE_BUILD_TYPE': 'Release'},
                            'core_command_example': command, 'driver_command_example': command}}}))
            self.assertIn('core_sha256', runner.release_runtime(context, build, provenance))
            with self.assertRaisesRegex(RuntimeError, 'Missing loaded core library identity'):
                runner.release_runtime({'build': 'Release', 'library_build_type': 'release'}, build, provenance)
            library.write_bytes(b'different binary')
            with self.assertRaisesRegex(RuntimeError, 'hash does not match'):
                runner.release_runtime(context, build, provenance)
            with self.assertRaisesRegex(RuntimeError, 'runtime is not Release'):
                runner.release_runtime({**context, 'library_build_type': 'debug'}, build)
            with self.assertRaisesRegex(RuntimeError, 'driver runtime is not Release'):
                runner.release_runtime({**context, 'build': 'Clang/Debug'}, build)

    def test_legacy_suite_checks_its_own_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Release\n')
            (root / 'compile_commands.json').write_text(json.dumps([
                {'file': '/repo/src/datamunge/a.cpp', 'command': 'c++ -O3 -DNDEBUG -c a.cpp'},
                {'file': '/repo/benchmarks/backend/main.cpp', 'command': 'c++ -O3 -DNDEBUG -c main.cpp'},
                {'file': '/repo/benchmarks/nn/main.cpp', 'command': 'c++ -O0 -c main.cpp'}]))
            self.assertTrue(runner.release_manifest(root)['ndebug_verified'])
            with self.assertRaisesRegex(RuntimeError, 'not an optimized Release'):
                runner.release_manifest(root, 'nn_bench')

    def test_median_comparison_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.json', Path(directory) / 'b.json'
            a.write_text(json.dumps({'benchmarks': [{'name': 'x', 'real_time': v, 'time_unit': 'ms'} for v in [1, 2, 90]]}))
            b.write_text(json.dumps({'benchmarks': [{'name': 'x', 'real_time': 1000, 'time_unit': 'us'}]}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                runner.compare(SimpleNamespace(base=str(a), new=str(b)))
            self.assertIn('x\t2\t1\t-50.00', out.getvalue())

    def test_incompatible_thread_counts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('a.json', 'b.json')]
            for count, path in enumerate(paths, 1):
                path.write_text(json.dumps({'backend_benchmark_metadata': {'thread_environment': {'OMP_NUM_THREADS': str(count)}}, 'benchmarks': []}))
            with self.assertRaisesRegex(RuntimeError, 'thread_environment'):
                runner.compare(SimpleNamespace(base=str(paths[0]), new=str(paths[1])))

    def test_measurement_errors_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'errors.json'
            p.write_text(json.dumps({'benchmarks': [{'name': 'x', 'error_occurred': True, 'error_message': 'bad fixture'}]}))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'benchmark errors'):
                runner.compare(SimpleNamespace(base=str(p), new=str(p)))

    def test_debug_core_rejected_even_with_release_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'compile_commands.json').write_text(json.dumps([
                {'file': '/repo/src/datamunge/a.cpp', 'command': 'c++ -O3 -DNDEBUG -O0 -c a.cpp'},
                {'file': '/repo/benchmarks/backend/main.cpp', 'command': 'c++ -O3 -DNDEBUG -c main.cpp'}]))
            with self.assertRaisesRegex(RuntimeError, 'not an optimized Release'):
                runner.release_manifest(p)

    def test_missing_benchmark_registration_fails_audit(self):
        from unittest.mock import patch
        with patch.object(runner.subprocess, 'check_output', return_value='BM_Unrelated/1\n'):
            with self.assertRaisesRegex(RuntimeError, 'Missing benchmark coverage'):
                runner.audit(Path('/unused-build'))

    def test_empty_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'empty.json'; p.write_text('{"benchmarks": []}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'No comparable cases'):
                runner.compare(SimpleNamespace(base=str(p), new=str(p)))



class HistoryTests(unittest.TestCase):
    def result(self, path, duration='0.1s', failed=False):
        metadata = {'revision': 'abc', 'release': {'configuration': {'CMAKE_BUILD_TYPE': 'Release'}},
                    'machine': 'test', 'thread_environment': {'OMP_NUM_THREADS': '1'},
                    'command': ['bench', '--benchmark_min_time=' + duration]}
        rows = [{'name': 'copy/1024', 'real_time': t, 'time_unit': 'us',
                 'run_type': 'iteration', 'peak_tensor_bytes': 4096} for t in (1, 2, 9)]
        rows.append({'name': 'copy/1024_mean', 'run_type': 'aggregate', 'real_time': 999, 'time_unit': 'us'})
        if failed: rows[0].update(error_occurred=True, error_message='bad fixture')
        path.write_text(json.dumps({'context': {'date': '2026-10-24T00:00:00Z'},
                                    'backend_benchmark_metadata': metadata, 'benchmarks': rows}))

    def test_archive_preserves_raw_data_and_median_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.json'; archive = root / 'history'
            self.result(source)
            with contextlib.redirect_stdout(io.StringIO()):
                first = runner.record_result(source, archive, 'baseline')
                self.assertEqual(first, runner.record_result(source, archive, 'baseline'))
            self.assertEqual((archive / 'runs' / (first + '.json')).read_bytes(), source.read_bytes())
            record = json.loads((archive / 'summaries' / (first + '.json')).read_text())
            self.assertEqual(record['cases']['copy/1024']['median_ns'], 2000)
            self.assertEqual(record['cases']['copy/1024']['repetitions'], 3)
            self.assertEqual(record['cases']['copy/1024']['counters']['peak_tensor_bytes'], 4096)
            self.assertEqual(record['kind'], 'calibrated')
            self.assertEqual(len(list((archive / 'summaries').glob('*.json'))), 1)

    def test_corrupted_archive_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.json'; archive = root / 'history'
            self.result(source)
            with contextlib.redirect_stdout(io.StringIO()): identity = runner.record_result(source, archive)
            (archive / 'runs' / (identity + '.json')).write_text('corrupt')
            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                runner.record_result(source, archive)

    def test_failed_run_is_not_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.json'; archive = Path(tmp) / 'history'
            self.result(source, failed=True)
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                runner.record_result(source, archive)
            self.assertFalse(archive.exists())

    def test_smoke_is_excluded_and_profiles_filter_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.json'; archive = Path(tmp) / 'history'
            self.result(source, duration='1x')
            with contextlib.redirect_stdout(io.StringIO()): runner.record_result(source, archive)
            args = SimpleNamespace(directory=archive, case='copy', profile=None, include_smoke=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out): runner.history(args)
            self.assertEqual(len(out.getvalue().splitlines()), 1)
            args.include_smoke = True
            out = io.StringIO()
            with contextlib.redirect_stdout(out): runner.history(args)
            self.assertEqual(len(out.getvalue().splitlines()), 2)
            args.profile = 'not-a-profile'
            out = io.StringIO()
            with contextlib.redirect_stdout(out): runner.history(args)
            self.assertEqual(len(out.getvalue().splitlines()), 1)

    def test_dates_are_normalized_before_ordering(self):
        a = runner.history_date({'date': '2026-10-24T08:00:00+02:00'})
        b = runner.history_date({'date': '2026-10-24T04:30:00-02:00'})
        self.assertLess(a, b)
        self.assertEqual(a, '2026-10-24T06:00:00+00:00')

    def test_invalid_timing_is_not_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.json'; self.result(source)
            data = json.loads(source.read_text()); data['benchmarks'][0]['real_time'] = float('nan')
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'invalid'):
                runner.record_result(source, Path(tmp) / 'history')


if __name__ == '__main__':
    unittest.main()
