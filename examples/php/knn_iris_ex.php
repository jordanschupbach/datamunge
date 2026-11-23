<?php
// CMD: just TARGET=knn_iris_ex run-php
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\KnnClassifier;
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

$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n\n";

$model = new KnnClassifier($iris, 'Species ~ Petal.Length + Petal.Width');
$model->printSummary();

echo "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n";
echo $model->confusionMatrix(), "\n";

echo "\nLeave-one-out misclassified rows:\n";
$fitted = $model->predict($iris);
$n = $iris->nrows();
$misclassified = 0;
for ($i = 0; $i < $n; $i++) {
    [$actual, $predicted] = [$iris->stringAt('Species', $i), $fitted[$i]];
    if ($predicted === $actual) {
        continue;
    }
    $misclassified++;
    echo "  row $i: Petal.Length=" . cs($iris->numericAt('Petal.Length', $i)) . ' Petal.Width='
        . cs($iris->numericAt('Petal.Width', $i)) . "  actual=$actual  predicted=$predicted\n";
}
echo "$misclassified of $n misclassified (" . cs(100.0 * $misclassified / $n) . "%)\n";

$model->plotDecisionRegions('Petal.Length', 'Petal.Width')->save('knn_iris_decision_regions_k5.svg');
echo "\nSaved knn_iris_decision_regions_k5.svg\n";

// k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
// every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
// underfit boundary). new KnnClassifier(data, formula, k, metric, weighted, standardize)
foreach ([1, 25] as $k) {
    $knn = new KnnClassifier($iris, 'Species ~ Petal.Length + Petal.Width', $k);
    $file = "knn_iris_decision_regions_k$k.svg";
    echo "\nk=" . str_pad((string) $k, 2) . ' leave-one-out accuracy: ' . cs($knn->trainingAccuracy() * 100) . "%\n";
    $knn->plotDecisionRegions('Petal.Length', 'Petal.Width')->save($file);
    echo "Saved $file\n";
}
