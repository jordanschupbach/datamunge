#!/usr/bin/env python3
"""Turn blas_compare JSON into markdown tables (one per op/dtype/thread count)."""
import argparse
import json
from collections import OrderedDict


def ratio(inhouse, libs, metric):
    """In-house speed over the best library's (> 1: in-house faster)."""
    if metric == 'ms':
        return min(libs) / min(inhouse) if min(inhouse) > 0 else None
    return max(inhouse) / max(libs) if max(libs) > 0 else None


def fmt(value, metric):
    """Times: three decimals, or three significant digits below 1 ms (small matrices)."""
    if metric == 'ms':
        return '%.3f' % value if value >= 1 else '%.3g' % value
    return '%.1f' % value


def table(rows, metric):
    backends = list(OrderedDict.fromkeys(r['backend'] for r in rows))
    cases = list(OrderedDict.fromkeys(r['case'] for r in rows))
    lookup = {(r['case'], r['backend']): r for r in rows}
    best_lib = [b for b in backends if not b.startswith('inhouse')]
    head = '| case | ' + ' | '.join(backends) + (' | in-house speed / best library |' if best_lib else ' |')
    lines = [head, '|' + '---|' * (len(backends) + 1 + bool(best_lib))]
    for c in cases:
        vals = []
        for b in backends:
            r = lookup.get((c, b))
            vals.append(fmt(r[metric], metric) if r else '')
        line = '| %s | %s' % (c, ' | '.join(vals))
        if best_lib:
            inhouse = [lookup[(c, b)][metric] for b in backends if b.startswith('inhouse') and (c, b) in lookup]
            libs = [lookup[(c, b)][metric] for b in best_lib if (c, b) in lookup]
            q = ratio(inhouse, libs, metric) if inhouse and libs else None
            line += ' | %.2f' % q if q is not None else ' |'
        lines.append(line + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json')
    parser.add_argument('--out', help='markdown file (default: stdout)')
    args = parser.parse_args()
    rows = json.load(open(args.json))
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r['op'], r['dtype'], r['threads']), []).append(r)
    # Op, then dtype, then thread count (blas_compare writes its processes' rows thread by thread).
    first = {}
    for key in groups:
        first.setdefault(('op', key[0]), len(first))
        first.setdefault(('dtype', key[1]), len(first))
    order = sorted(groups, key=lambda k: (first[('op', k[0])], first[('dtype', k[1])], k[2]))
    parts = []
    for op, dtype, threads in order:
        g = groups[(op, dtype, threads)]
        metric = 'gflops' if op in ('gemm', 'gemv') else 'ms' if op == 'linalg' else 'gbps'
        unit = {'gflops': 'GFLOP/s', 'ms': 'ms per call, lower is better', 'gbps': 'GB/s'}[metric]
        where = 'GPU' if threads == 0 else '%d thread%s' % (threads, '' if threads == 1 else 's')
        how = ('median of %d interleaved rounds' % max(r.get('rounds', 1) for r in g) if op == 'linalg'
               else 'median of >= 5 calls')
        parts.append('### %s %s, %s (%s, %s)\n\n%s\n' % (op, dtype, where, unit, how, table(g, metric)))
    text = '\n'.join(parts)
    if args.out:
        open(args.out, 'w').write(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
