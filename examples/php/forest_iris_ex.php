<?php
// CMD: just TARGET=forest_iris_ex run-php
// Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\RandomForestClassifier;
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

$model = new RandomForestClassifier($iris, 'Species ~ Petal.Length + Petal.Width');
$model->printSummary();

echo "\nConfusion matrix (rows = actual, cols = predicted):\n";
echo $model->confusionMatrix(), "\n";

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

$model->plotClassification($iris, 'Petal.Length', 'Petal.Width')->save('forest_iris_classification.svg');
$model->plotDecisionRegions('Petal.Length', 'Petal.Width')->save('forest_iris_decision_regions.svg');
echo "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n";

// A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
// are added -- the defining random forest effect a single decision tree cannot demonstrate.
// new RandomForestClassifier(data, formula, nTrees, maxDepth, minSamplesSplit, minSamplesLeaf,
//                            maxFeatures, criterion, bootstrap, sampleFraction, seed)
$smallForest = new RandomForestClassifier($iris, 'Species ~ Petal.Length + Petal.Width', 5);
echo "\n5-tree forest:   training accuracy=" . cs($smallForest->trainingAccuracy() * 100) . '%  OOB accuracy='
    . cs($smallForest->oobAccuracy() * 100) . "%\n";
echo '100-tree forest: training accuracy=' . cs($model->trainingAccuracy() * 100) . '%  OOB accuracy='
    . cs($model->oobAccuracy() * 100) . "%\n";
$smallForest->plotDecisionRegions('Petal.Length', 'Petal.Width')->save('forest_iris_decision_regions_5trees.svg');
echo "Saved forest_iris_decision_regions_5trees.svg\n";
