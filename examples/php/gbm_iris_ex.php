<?php
// CMD: just TARGET=gbm_iris_ex run-php
// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\GbmClassifier;
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

/** The last element of a list (C#'s [^1]). */
function last(array $values): float
{
    return $values[count($values) - 1];
}

$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n\n";

$model = new GbmClassifier($iris, 'Species ~ Petal.Length + Petal.Width');
$model->printSummary();

echo "\nConfusion matrix (rows = actual, cols = predicted):\n{$model->confusionMatrix()}\n";

echo "\nMisclassified rows:\n";
$predictions = $model->predict($iris);
$n = $iris->nrows();
$misclassified = 0;
for ($i = 0; $i < $n; $i++) {
    [$actual, $predicted] = [$iris->stringAt('Species', $i), $predictions[$i]];
    if ($predicted === $actual) {
        continue;
    }
    $misclassified++;
    echo "  row $i: Petal.Length=" . cs($iris->numericAt('Petal.Length', $i)) . ' Petal.Width='
        . cs($iris->numericAt('Petal.Width', $i)) . "  actual=$actual  predicted=$predicted\n";
}
echo "$misclassified of $n misclassified (" . cs(100.0 * $misclassified / $n) . "%)\n";

$model->plotTrainingDeviance()->save('gbm_iris_training_deviance.svg');
$model->plotDecisionRegions('Petal.Length', 'Petal.Width')->save('gbm_iris_decision_regions.svg');
echo "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n";

// A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
// at the training loss, gradually sharpening the decision boundary.
// new GbmClassifier(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
//                   minSamplesLeaf, subsample, seed)
$few = new GbmClassifier($iris, 'Species ~ Petal.Length + Petal.Width', 5);
echo "\n5-round ensemble:   training accuracy=" . cs(100 * $few->trainingAccuracy()) . '%  deviance='
    . cs(last($few->trainingDeviance())) . "\n";
echo '100-round ensemble: training accuracy=' . cs(100 * $model->trainingAccuracy()) . '%  deviance='
    . cs(last($model->trainingDeviance())) . "\n";
$few->plotDecisionRegions('Petal.Length', 'Petal.Width')->save('gbm_iris_decision_regions_5rounds.svg');
echo "Saved gbm_iris_decision_regions_5rounds.svg\n";
