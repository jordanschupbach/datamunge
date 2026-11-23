<?php
// CMD: just TARGET=tensor_ex run-php
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\Tensor;
use function Datamunge\Data\iris;

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

function shapeString(Tensor $tensor): string
{
    return '[' . implode(', ', $tensor->shape()) . ']';
}

echo "=================== Construction ===================\n";
echo 'zeros([2,3]): ' . Tensor::zeros([2, 3]) . "\n";
echo 'eye(3): ' . Tensor::eye(3) . "\n";

$r = Tensor::arange(0.0, 12.0, 1.0)->reshape([3, 4]);
echo "arange(0,12).reshape([3,4]): $r\n";

echo "\n=================== Shape ops ===================\n";
echo 'transpose -> shape ' . shapeString($r->transpose()) . "\n";
echo "slice(axis=1, start=1, stop=3): {$r->slice(1, 1, 3)}\n";

echo "\n=================== Broadcasting arithmetic ===================\n";
$col = Tensor::fromValues([3, 1], [1.0, 2.0, 3.0]);
$row = Tensor::fromValues([1, 4], [10.0, 20.0, 30.0, 40.0]);
echo "(3,1) + (1,4) -> {$col->add($row)}\n";

echo "\n=================== Reductions ===================\n";
echo 'r.sum() = ' . cs($r->sum()) . ', r.mean() = ' . cs($r->mean()) . "\n";
echo "column means (axis=0): {$r->meanAxis(0)}\n";

echo "\n=================== Linear algebra ===================\n";
$a = Tensor::fromValues([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
$b = Tensor::fromValues([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
echo "matmul(2x3, 3x2) -> {$a->matmul($b)}\n";
$v1 = Tensor::fromValues([3], [1.0, 2.0, 3.0]);
$v2 = Tensor::fromValues([3], [4.0, 5.0, 6.0]);
echo 'dot([1,2,3], [4,5,6]) = ' . cs($v1->dot($v2)) . "\n";
echo "outer(v1, v2) -> {$v1->outer($v2)}\n";

echo "\n=================== Comparisons & masks ===================\n";
$mask = $r->greaterEqual(Tensor::full([3, 4], 6.0));
echo "r >= 6 -> $mask\n";
echo 'count(r >= 6) = ' . cs($mask->sum()) . "\n";

echo "\n=================== A real dataset as a Tensor ===================\n";
$iris = iris();
$n = $iris->nrows();
$columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];
$flat = [];
for ($i = 0; $i < $n; $i++) {
    foreach ($columns as $column) {
        $flat[] = $iris->numericAt($column, $i);
    }
}
$x = Tensor::fromValues([$n, 4], $flat);
echo 'iris feature tensor shape: ' . shapeString($x) . "\n";

$featureMeans = $x->meanAxis(0);
$centered = $x->subtract($featureMeans->reshape([1, 4]));
$scatter = $centered->transpose()->matmul($centered);
echo "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): $featureMeans\n";
echo "(X-mean)^T (X-mean) [4x4 scatter matrix]: {$scatter->toString(16)}\n";

$species = [];
for ($i = 0; $i < $n; $i++) {
    $species[] = $iris->stringAt('Species', $i);
}
$speciesTensor = Tensor::fromStringValues([$n], $species);
$setosaMask = $speciesTensor->equal(Tensor::fromStringValues([1], ['setosa']));
echo 'setosa count = ' . cs($setosaMask->sum()) . " (of $n rows)\n";
