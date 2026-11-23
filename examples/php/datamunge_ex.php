<?php
// CMD: just TARGET=datamunge_ex run-php
// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\Callback;
use function Datamunge\{call_with_callback, hello, make_dpair, make_dvector, map_dvector_with_callback, sum_dpair,
    sum_dvector};

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

hello();

$v = make_dvector(1.0, 2.0, 3.0);
echo "sum_dvector:\t" . cs(sum_dvector($v)) . "\n";

$p = make_dpair(1.25, 2.75);
echo "sum_dpair:\t" . cs(sum_dpair($p)) . "\n";

$cb = new Callback();
echo "call_with_callback(3.0):\t" . cs(call_with_callback(3.0, $cb)) . "\n";
$v2 = map_dvector_with_callback(make_dvector(1.0, 2.0, 3.0), $cb);
echo "sum_dvector(map_dvector_with_callback(1,2,3)):\t" . cs(sum_dvector($v2)) . "\n";
