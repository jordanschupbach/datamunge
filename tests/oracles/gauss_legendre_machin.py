"""Regenerate/check the independent AGM reference using only Decimal Machin series.

Run `python3 tests/oracles/gauss_legendre_machin.py --write` to regenerate the
adjacent C++ fixture, or `--check` (the default) to verify reproducibility.
No AGM recurrence is used. Fixed legacy approximations are compared with the
independently calculated real constant to separate rounding from convergence.
"""
from argparse import ArgumentParser
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from pathlib import Path
import json


def atan_inverse(n):
    x = Decimal(1) / n
    term = total = x
    k = 1
    while True:
        term *= -x * x
        updated = total + term / (2 * k + 1)
        if updated == total:
            return total
        total = updated
        k += 1


def reference():
    with localcontext() as context:
        context.prec = 150
        context.rounding = ROUND_HALF_EVEN
        pi = 16 * atan_inverse(5) - 4 * atan_inverse(239)
        estimates = [float.fromhex(x) for x in (
            '0x1.91fe8039be947p+1', '0x1.921fb5346bae9p+1', '0x1.921fb54442d1ap+1')]
        errors = [abs(Decimal.from_float(x) - pi) for x in estimates]
        header = ('// Independent Decimal150 Machin oracle; regenerate: python3 tests/oracles/gauss_legendre_machin.py --write\n'
                  '#pragma once\nnamespace gauss_legendre_oracle {\n'
                  'inline constexpr long double pi=' + str(pi) + 'L;\n'
                  'inline constexpr double rounded=' + float(pi).hex() + ';\n'
                  'inline constexpr long double true_error[] = {' + ','.join(str(e) + 'L' for e in errors) + '};\n}\n')
        return header, {'method': 'Independent Decimal150 Machin16atan(1/5)-4atan(1/239), alternating power series until decimal sum unchanged',
                        'pi': str(pi), 'rounded_double_hex': float(pi).hex(), 'true_errors': [str(e) for e in errors]}


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--json', type=Path, help='Optional development evidence destination.')
    args = parser.parse_args()
    header, evidence = reference()
    target = Path(__file__).with_suffix('.hpp')
    if args.write:
        target.write_text(header)
    else:
        assert target.read_text() == header, 'Machin fixture differs; regenerate with --write'
    if args.json:
        args.json.write_text(json.dumps(evidence, indent=2) + '\n')
    print('Independent Decimal150 Machin reference: fixture verified')


if __name__ == '__main__':
    main()
