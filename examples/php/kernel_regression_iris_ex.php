<?php
// CMD: just TARGET=kernel_regression_iris_ex run-php
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\KernelRegression;
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

const FORMULA = 'Petal.Length ~ Petal.Width';

$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n";
echo 'formula: ' . FORMULA . "\n\n";

$model = new KernelRegression($iris, FORMULA);  // bandwidth auto-selected via leave-one-out CV
$model->printSummary();

$model->plotFit($iris)->save('kernel_regression_iris_fit.svg');
$model->plotCvCurve()->save('kernel_regression_iris_cv.svg');
echo "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n";

// A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
// bias-variance story. new KernelRegression(data, formula, kernel, bandwidth, nBandwidth, standardize)
$small = new KernelRegression($iris, FORMULA, 'gaussian', 0.05);
echo "\nbandwidth=0.05 (too small): LOO R-squared=" . cs($small->rSquared()) . '  LOO RMSE=' . cs($small->rmse()) . "\n";
$small->plotFit($iris)->save('kernel_regression_iris_fit_small_bandwidth.svg');

$large = new KernelRegression($iris, FORMULA, 'gaussian', 5.0);
echo 'bandwidth=5.0 (too large):  LOO R-squared=' . cs($large->rSquared()) . '  LOO RMSE=' . cs($large->rmse()) . "\n";
$large->plotFit($iris)->save('kernel_regression_iris_fit_large_bandwidth.svg');

echo 'bandwidth=' . cs($model->bandwidth()) . ' (CV-selected): LOO R-squared=' . cs($model->rSquared())
    . '  LOO RMSE=' . cs($model->rmse()) . "\n";
echo "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n";
