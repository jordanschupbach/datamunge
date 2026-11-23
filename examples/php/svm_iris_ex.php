<?php
// CMD: just TARGET=svm_iris_ex run-php
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Svm};
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

const FORMULA = 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width';
$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n\n";

echo "=== RBF kernel (default) ===\n";
$rbfModel = new Svm($iris, FORMULA);
$rbfModel->printSummary();
echo "\nConfusion matrix (rows = actual, cols = predicted):\n{$rbfModel->confusionMatrix()}\n";

echo "\n=== Linear kernel, for comparison ===\n";
$linearModel = new Svm($iris, FORMULA, 'linear');
echo 'Training accuracy: ' . cs($linearModel->trainingAccuracy() * 100) . "%\n";
echo "Support vectors: {$linearModel->numSupportVectors()}\n";

$newdata = new DataFrame();
$newdata->addNumericColumn('Sepal.Length', [5.1, 6.0, 6.5, 6.2]);
$newdata->addNumericColumn('Sepal.Width', [3.5, 2.7, 3.0, 2.8]);
$newdata->addNumericColumn('Petal.Length', [1.4, 4.5, 5.5, 4.8]);
$newdata->addNumericColumn('Petal.Width', [0.2, 1.5, 2.0, 1.8]);
echo "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n{$rbfModel->predictFrame($newdata)}\n";
