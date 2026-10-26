"""Cross-framework pairing contracts; TensorFlow is not required for these tests."""
from pathlib import Path
import sys
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_tensorflow as driver


def observations():
    native, tensorflow = {'benchmarks': []}, {'benchmarks': []}
    for op in driver.OPERATIONS:
        for rows in driver.CASES[op]:
            for f64 in (0, 1):
                suffix = f'{op}/rows:{rows}/f64:{f64}/real_time'
                for value in (1, 2, 90):
                    native['benchmarks'].append({'name': 'BM_TensorFlowMatched/' + suffix,
                                                 'real_time': value, 'time_unit': 'us'})
                for mode in ('eager', 'graph'):
                    tensorflow['benchmarks'].append({'name': f'TensorFlow/{mode}/' + suffix,
                                                     'real_time': 1000})
    return native, tensorflow


class TensorFlowComparisonTests(unittest.TestCase):
    def test_catalog_spans_components_and_retains_original_cases(self):
        self.assertGreaterEqual(len(driver.OPERATIONS), 30)
        for operation in ('Conv2d', 'ConvInputGradient', 'Attention', 'LayerNorm', 'BatchNorm',
                          'FftPower', 'Solve', 'Cholesky', 'SoftmaxGradient', 'MatmulGradient'):
            self.assertIn(operation, driver.CASES)
        for sizes in driver.CASES.values():
            self.assertEqual(len(sizes), len(set(sizes)))
            self.assertTrue(all(size > 0 for size in sizes))
        self.assertEqual(len(driver.selected_cases()), sum(len(s) * 2 for s in driver.CASES.values()))
        self.assertEqual(len(driver.selected_cases(r'^Conv2d/rows:16/')), 2)
        self.assertEqual(len(driver.selected_cases(r'^Matmul/')), 6)

    def test_native_smoke_selects_every_comparison_family(self):
        for operation, sizes in driver.CASES.items():
            for f64 in (0, 1):
                self.assertRegex(f'BM_TensorFlowMatched/{operation}/rows:{sizes[0]}/f64:{f64}/real_time',
                                 driver.backend.SMOKE)

    def test_debug_preflight_fails_before_native_execution(self):
        args = SimpleNamespace(build_dir=Path('/unused'))
        with patch.object(driver.backend, 'release_manifest', side_effect=RuntimeError('not Release')), \
             patch.object(driver.subprocess, 'run') as execute:
            with self.assertRaisesRegex(RuntimeError, 'not Release'):
                driver.native_case(args, 'fixture', {})
            execute.assert_not_called()

    def test_selection_and_filtered_comparison(self):
        selection = driver.selected_cases(r'(SumAll|SquareGradient)/rows:8192/f64:0/')
        self.assertEqual(len(selection), 2)
        native, tensorflow = observations()
        native['benchmarks'] = [r for r in native['benchmarks'] if r['name'] in selection]
        tf_names = {name.replace('BM_TensorFlowMatched/', f'TensorFlow/{mode}/', 1)
                    for name in selection for mode in ('eager', 'graph')}
        tensorflow['benchmarks'] = [r for r in tensorflow['benchmarks'] if r['name'] in tf_names]
        self.assertEqual(len(driver.compare(native, tensorflow, selection)), 2)
        with self.assertRaisesRegex(RuntimeError, 'requested case selection'):
            driver.compare(native, tensorflow, driver.selected_cases())
        with self.assertRaisesRegex(RuntimeError, 'matched no fixtures'):
            driver.selected_cases('not-an-operation')

    def test_pairing_medians_and_unit_conversion(self):
        native, tensorflow = observations()
        native['benchmarks'].append({'name': 'unrelated', 'real_time': 100, 'time_unit': 'ns'})
        native['benchmarks'].append({'name': 'aggregate', 'run_type': 'aggregate'})
        rows = driver.compare(native, tensorflow)
        self.assertEqual(len(rows), len(driver.selected_cases()))
        self.assertTrue(all(row['native_over_eager'] == 2 and row['native_ns'] == 2000 for row in rows))

    def test_missing_graph_case_is_rejected(self):
        native, tensorflow = observations()
        tensorflow['benchmarks'].pop()
        with self.assertRaisesRegex(RuntimeError, 'Missing TensorFlow case'):
            driver.compare(native, tensorflow)

    def test_native_failure_is_rejected(self):
        native, tensorflow = observations()
        native['benchmarks'][0]['error_occurred'] = True
        with self.assertRaisesRegex(RuntimeError, 'Native reference/timing failure'):
            driver.compare(native, tensorflow)

    def test_nonfinite_tensorflow_timing_is_rejected(self):
        native, tensorflow = observations()
        tensorflow['benchmarks'][0]['real_time'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Invalid TensorFlow timing'):
            driver.compare(native, tensorflow)

    def test_paired_context_allows_load_drift_but_rejects_build_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / 'backend_bench').write_bytes(b'fixture executable')
            args = SimpleNamespace(build_dir=build, min_time=0.1, repetitions=3, label='test')
            context = {'date': 'first', 'load_avg': [1, 2, 3], 'build': 'Release'}
            def execute(command, **kwargs):
                output = Path(next(value.split('=', 1)[1] for value in command if value.startswith('--benchmark_out=')))
                output.write_text(json.dumps({'context': dict(context), 'benchmarks': [
                    {'name': 'case', 'real_time': 100, 'reference_checked': 1} for _ in range(3)]}))
            native = {}
            with patch.object(driver.subprocess, 'run', side_effect=execute), \
                 patch.object(driver.backend, 'release_manifest', return_value={}), \
                 patch.object(driver.backend, 'release_runtime', return_value={}), \
                 patch.object(driver.backend, 'revision', return_value={}), \
                 patch.object(driver.platform, 'platform', return_value='test'), \
                 patch.object(driver.platform, 'machine', return_value='test'):
                driver.native_case(args, 'fixture', native)
                context.update(date='second', load_avg=[4, 5, 6])
                driver.native_case(args, 'fixture', native)
                self.assertEqual(len(native['benchmarks']), 6)
                context['build'] = 'Debug'
                with self.assertRaisesRegex(RuntimeError, 'context changed'):
                    driver.native_case(args, 'fixture', native)


if __name__ == '__main__':
    unittest.main()
