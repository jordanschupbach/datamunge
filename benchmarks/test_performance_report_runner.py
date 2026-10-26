from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import performance_report as report


class PerformanceReportTests(unittest.TestCase):
    def row(self, name):
        return {'case':'BM_TensorFlowMatched/'+name, 'native_ns':2000, 'eager_ns':1000,
                'graph_ns':4000, 'native_over_eager':2, 'native_over_graph':.5}

    def test_preserves_manual_sections_and_includes_every_run_and_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); destination=root/'PERFORMANCE.md'; results=root/'results';results.mkdir()
            destination.write_text('# Manual plan\n\nAccepted constraints.\n')
            (results/'before.comparison.json').write_text(json.dumps([self.row('Before/rows:1/f64:0/real_time')]))
            report.update_report(destination,results,root)
            first=destination.read_text()
            self.assertIn('Accepted constraints.',first)
            self.assertIn('Before/rows:1',first)
            report.update_report(destination,results,root)
            self.assertEqual(first,destination.read_text())
            (results/'after.comparison.json').write_text(json.dumps([self.row('After/rows:2/f64:1/real_time')]))
            report.update_report(destination,results,root)
            final=destination.read_text()
            self.assertIn('2 runs; 2 per-case comparisons',final)
            self.assertIn('Before/rows:1',final);self.assertIn('After/rows:2',final)
            self.assertEqual(final.count(report.START),1)

    def test_malformed_markers_do_not_overwrite_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);destination=root/'PERFORMANCE.md';destination.write_text(report.START)
            with self.assertRaisesRegex(ValueError,'markers'):
                report.update_report(destination,root,root)
            self.assertEqual(destination.read_text(),report.START)

    def test_invalid_data_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'broken.comparison.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'comparison'):
                report.update_report(root/'PERFORMANCE.md',root,root)

    def test_reusing_or_removing_output_filename_keeps_historical_comparisons(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'latest.comparison.json'; destination=root/'PERFORMANCE.md'
            raw=root/'latest.json'
            raw.write_text(json.dumps({'backend_benchmark_metadata':{'core_sources_sha256':'old'}}))
            source.write_text(json.dumps([self.row('Old/rows:1/f64:0/real_time')]))
            report.update_report(destination,root,root)
            raw.write_text(json.dumps({'backend_benchmark_metadata':{'core_sources_sha256':'new'}}))
            source.write_text(json.dumps([self.row('New/rows:2/f64:1/real_time')]))
            report.update_report(destination,root,root)
            self.assertIn('Old/rows:1',destination.read_text())
            self.assertIn('New/rows:2',destination.read_text())
            snapshots=[json.loads(p.read_text()) for p in (root/'comparisons').glob('*.json')]
            self.assertEqual({s['provenance']['core_sources_sha256'] for s in snapshots},{'old','new'})
            source.unlink()
            report.update_report(destination,root,root)
            self.assertIn('2 runs; 2 per-case comparisons',destination.read_text())
