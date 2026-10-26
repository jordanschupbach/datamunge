#!/usr/bin/env python3
"""Maintain every TensorFlow/native result table inside the single performance report."""
import argparse
import html
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- tensorflow-comparisons:start -->'
END = '<!-- tensorflow-comparisons:end -->'


def comparison_records(directory, root=ROOT):
    directory = Path(directory)
    archive = directory / 'comparisons'
    for source in sorted(directory.glob('*.comparison.json')):
        rows = json.loads(source.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'Invalid comparison table: {source}')
        canonical = json.dumps(sorted(rows, key=lambda row: row['case']), sort_keys=True,
                               separators=(',', ':'), allow_nan=False)
        identity = hashlib.sha256(canonical.encode()).hexdigest()
        target = archive / (identity + '.json')
        raw = source.with_name(source.name.replace('.comparison.json', '.json'))
        observation = json.loads(raw.read_text()) if raw.exists() else {}
        record = json.loads(target.read_text()) if target.exists() else None
        if record is None or 'provenance' not in record:
            archive.mkdir(exist_ok=True)
            record = record or {'id': identity, 'label': source.stem.removesuffix('.comparison'),
                      'source': source.resolve().relative_to(Path(root).resolve()).as_posix(),
                      'context': observation.get('context', {}), 'rows': rows}
            record['provenance'] = observation.get('backend_benchmark_metadata', {})
            with tempfile.NamedTemporaryFile(mode='w', dir=archive, delete=False) as stream:
                snapshot = Path(stream.name)
                stream.write(json.dumps(record, indent=2) + '\n')
            snapshot.replace(target)
    records = []
    for path in archive.glob('*.json') if archive.exists() else ():
        data = json.loads(path.read_text())
        canonical = json.dumps(sorted(data['rows'], key=lambda row: row['case']), sort_keys=True,
                               separators=(',', ':'), allow_nan=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != data['id']:
            raise ValueError(f'Comparison archive integrity mismatch: {path}')
        records.append((path, data))
    return sorted(records, key=lambda item: (item[1]['label'], item[1]['id']))


def render_comparisons(directory, root=ROOT):
    files = comparison_records(directory, root)
    text = ['<a id="tensorflow-comparisons"></a>', '',
            '## All TensorFlow / in-house comparison tables', '',
            'Every archived comparison is included below, including superseded and exploratory runs. '
            'Use the dated results above for conclusions; do not combine different builds, fixture sets '
            'or noisy historical runs into one score. Ratios above 1 mean native took longer. '
            'Times are medians in microseconds; CV columns are percentages. '
            'Immutable snapshots retain older comparisons even when default output filenames are reused.', '']
    count = 0
    for path, record in files:
        rows = record['rows']
        label = html.escape(record['label'])
        relative = path.resolve().relative_to(Path(root).resolve()).as_posix()
        text.extend([f'<details><summary>{label} — {len(rows)} cases — {record["id"][:12]}</summary>', '',
                     f'Immutable data: [JSON]({relative}). Run date: `{record["context"].get("date", "unrecorded")}`.', '',
                     '| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |',
                     '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for row in sorted(rows, key=lambda r: r['case']):
            numbers = [row['native_ns']/1000, row['eager_ns']/1000, row['graph_ns']/1000,
                       row['native_over_eager'], row['native_over_graph'],
                       row.get('native_cv_percent'), row.get('eager_cv_percent'), row.get('graph_cv_percent')]
            case = row['case'].removeprefix('BM_TensorFlowMatched/').replace('|', '\\|')
            text.append('| `' + case + '` | ' + ' | '.join('—' if v is None else f'{v:.6g}' for v in numbers) + ' |')
        text.extend(['', '</details>', ''])
        count += len(rows)
    text.insert(5, f'**{len(files)} runs; {count} per-case comparisons.**')
    return '\n'.join(text)


def update_report(report=ROOT/'PERFORMANCE.md', directory=ROOT/'benchmarks/results/tensorflow', root=ROOT):
    report = Path(report)
    existing = report.read_text() if report.exists() else '# Performance plan and comparisons\n\n'
    content = render_comparisons(directory, root)
    block = START + '\n\n' + content + '\n' + END
    if START in existing or END in existing:
        if existing.count(START) != 1 or existing.count(END) != 1 or existing.index(START) >= existing.index(END):
            raise ValueError('Performance report comparison markers are malformed')
        begin, rest = existing.split(START)
        _, suffix = rest.split(END)
        updated = begin + block + suffix
    else:
        updated = existing.rstrip() + '\n\n' + block + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=report.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(updated)
    temporary.replace(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'PERFORMANCE.md')
    parser.add_argument('--results-dir', type=Path, default=ROOT/'benchmarks/results/tensorflow')
    args = parser.parse_args()
    print(update_report(args.report, args.results_dir))
