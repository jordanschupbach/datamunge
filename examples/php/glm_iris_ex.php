<?php
// CMD: just TARGET=glm_iris_ex run-php
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Glm};
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
$rows = range(0, $iris->nrows() - 1);

// Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
// the other two on these predictors, which sends logistic regression's coefficients toward
// +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
// binary split for a demo.
$kept = array_values(array_filter($rows,
    fn(int $i): bool => in_array($iris->stringAt('Species', $i), ['versicolor', 'virginica'], true)));
$isVirginica = array_map(fn(int $i): float => $iris->stringAt('Species', $i) === 'virginica' ? 1.0 : 0.0, $kept);
$petalLength = array_map(fn(int $i): float => $iris->numericAt('Petal.Length', $i), $kept);
$petalWidth = array_map(fn(int $i): float => $iris->numericAt('Petal.Width', $i), $kept);

$sub = new DataFrame();
$sub->addNumericColumn('Petal.Length', $petalLength);
$sub->addNumericColumn('Petal.Width', $petalWidth);
$sub->addNumericColumn('is_virginica', $isVirginica);

echo "=================== Logistic regression (binomial, logit link) ===================\n";
$logit = new Glm($sub, 'is_virginica ~ Petal.Length + Petal.Width', 'binomial');
$logit->printSummary();

$fitted = $logit->fittedValues();
$n = count($isVirginica);
$correct = count(array_filter(range(0, $n - 1), fn(int $i): bool => ($fitted[$i] >= 0.5) === ($isVirginica[$i] >= 0.5)));
echo "\nResubstitution accuracy at 0.5 threshold: " . cs(100.0 * $correct / $n) . "%\n";

$logit->saveDiagnosticPlots('glm_logistic_iris');
echo "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n";

// Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
// classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
$gridN = 100;
$widthMean = array_sum($petalWidth) / count($petalWidth);
[$plMin, $plMax] = [min($petalLength) - 0.3, max($petalLength) + 0.3];
$grid = new DataFrame();
$grid->addNumericColumn('Petal.Length', array_map(fn(int $i): float => $plMin + ($plMax - $plMin) * $i / ($gridN - 1),
    range(0, $gridN - 1)));
$grid->addNumericColumn('Petal.Width', array_fill(0, $gridN, $widthMean));
$curveFrame = $logit->predictFrame($grid, 'confidence');
echo "\nPredicted-probability curve (first 5 rows):\n";
echo $curveFrame->toString(5), "\n";

// Poisson regression, for contrast: same IRLS engine, different family/link.
echo "\n=================== Poisson regression (log link) ===================\n";
$countData = new DataFrame();
$countData->addNumericColumn('Sepal.Width', array_map(fn(int $i): float => $iris->numericAt('Sepal.Width', $i), $rows));
$countData->addNumericColumn('Petal.Length', array_map(fn(int $i): float => $iris->numericAt('Petal.Length', $i), $rows));
$countData->addNumericColumn('count', array_map(fn(int $i): float => floor($iris->numericAt('Sepal.Length', $i) + 0.5), $rows));
(new Glm($countData, 'count ~ Sepal.Width + Petal.Length', 'poisson'))->printSummary();
