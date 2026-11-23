<?php
// CMD: just TARGET=stl_ex run-php
// C++ std::vector and std::pair values in PHP: vectors are lists, pairs are arrays of two.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use function Datamunge\{make_dpair, sum_dpair};

/** A double the way C# prints it (shortest round-trip, e.g. 6, 0.1, 1E-05). */
function cs(float $x): string
{
    if (is_nan($x)) {
        return 'NaN';
    }
    if (is_infinite($x)) {
        return $x > 0 ? '∞' : '-∞';
    }
    if ($x == 0.0) {
        return fdiv(1.0, $x) < 0 ? '-0' : '0';
    }
    for ($p = 0; $p < 17; $p++) {
        $s = sprintf("%.{$p}e", $x);
        if ((float) $s === $x) {
            break;
        }
    }
    [$mant, $exp] = explode('e', $s);
    $exp = (int) $exp;
    $sign = $mant[0] === '-' ? '-' : '';
    $digits = rtrim(str_replace(['-', '.'], '', $mant), '0');
    if ($exp < -5 || $exp >= 15) {
        $body = $digits[0] . (strlen($digits) > 1 ? '.' . substr($digits, 1) : '')
            . 'E' . ($exp < 0 ? '-' : '+') . sprintf('%02d', abs($exp));
    } elseif ($exp < 0) {
        $body = '0.' . str_repeat('0', -$exp - 1) . $digits;
    } elseif (strlen($digits) <= $exp + 1) {
        $body = $digits . str_repeat('0', $exp + 1 - strlen($digits));
    } else {
        $body = substr($digits, 0, $exp + 1) . '.' . substr($digits, $exp + 1);
    }
    return $sign . $body;
}

const N = 100;
$v = array_map(fn(int $i): float => $i * 1.5, range(0, N - 1));
foreach ($v as $x) {
    echo cs($x), "\n";
}
$v2 = array_map(fn(int $i): int => (int) floor($i * 1.5), range(0, N - 1));
foreach ($v2 as $x) {
    echo $x, "\n";
}

$p = [3, 4];
echo "p: ($p[0], $p[1])\n";

$p2 = [10, 20];
echo "p2: ($p2[0], $p2[1])\n";

// A C++ std::pair<double,double> comes back as an array of two floats.
$p3 = make_dpair(1.25, 2.75);
echo 'make_dpair(1.25, 2.75): (' . cs($p3[0]) . ', ' . cs($p3[1]) . '), sum_dpair = ' . cs(sum_dpair($p3)) . "\n";
