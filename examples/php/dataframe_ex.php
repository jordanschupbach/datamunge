<?php
// CMD: just TARGET=dataframe_ex run-php
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\DataFrame;

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

$sales = new DataFrame();
$sales->addStringColumn('region', ['west', 'west', 'east', 'south', 'south', 'south']);
$sales->addStringColumn('product', ['widget', 'widget', 'widget', 'gizmo', 'gizmo', 'gizmo']);
$sales->addNumericColumn('sales', [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1]);
$sales->addStringColumn('quarter', ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', ''], [1, 1, 1, 1, 1, 0]);

echo "raw data\n$sales\n\n";

$cleaned = $sales->dropDuplicates(['region', 'product', 'sales', 'quarter']);
$cleaned->fillNullString('quarter', 'unknown');
$cleaned->fillNullNumeric('sales', 0.0);
echo "after drop_duplicates + fill_null\n$cleaned\n\n";

$selected = $cleaned->select(['region', 'sales', 'quarter'])->sortBy('sales', false);
echo "selected + sorted\n$selected\n\n";

$grouped = $cleaned->groupBySum(['region'], ['sales'])->sortBy('sales', false);
echo "group_by_sum(region)\n$grouped\n\n";

$targets = new DataFrame();
$targets->addStringColumn('region', ['west', 'east', 'south']);
$targets->addNumericColumn('target', [18.0, 12.0, 25.0]);
echo "joined with targets\n{$grouped->join($targets, 'region', 'region', 'left')}\n\n";

$shape = $cleaned->shape();
echo "shape = ($shape[0], $shape[1])\n";
echo "sales count = {$cleaned->numericCount('sales')}\n";
echo "sales nulls = {$cleaned->numericNullCount('sales')}\n";
echo 'sales sum = ' . cs($cleaned->numericSum('sales')) . "\n";
echo 'sales mean = ' . cs($cleaned->numericMean('sales')) . "\n";
