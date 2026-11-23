<?php
// CMD: just TARGET=naive_bayes_penguins_ex run-php
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\NaiveBayesClassifier;
use function Datamunge\Data\penguins;

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

// Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
// when fitting, matching every other formula-based model in this library.
$penguins = penguins();
echo "penguins: {$penguins->nrows()} rows x {$penguins->ncols()} cols\n\n";

// A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
// predictors in one formula -- each modeled independently given the class, per the naive
// Bayes assumption.
$model = new NaiveBayesClassifier($penguins, 'species ~ bill_length_mm + bill_depth_mm + island + sex');
$model->printSummary();

echo "\nConfusion matrix (rows = actual, cols = predicted):\n{$model->confusionMatrix()}\n";

echo "\nMisclassified rows:\n";
$predictions = $model->predict($penguins);
$misclassified = 0;
$columns = ['species', 'bill_length_mm', 'bill_depth_mm', 'island', 'sex'];
for ($i = 0; $i < $penguins->nrows(); $i++) {
    if (array_filter($columns, fn(string $c): bool => $penguins->isNull($c, $i)) !== []) {
        continue;
    }
    [$actual, $predicted] = [$penguins->stringAt('species', $i), $predictions[$i]];
    if ($predicted === $actual) {
        continue;
    }
    $misclassified++;
    echo "  row $i: bill_length=" . cs($penguins->numericAt('bill_length_mm', $i)) . ' bill_depth='
        . cs($penguins->numericAt('bill_depth_mm', $i)) . ' '
        . "island={$penguins->stringAt('island', $i)} sex={$penguins->stringAt('sex', $i)}  actual=$actual  predicted=$predicted\n";
}
echo "$misclassified misclassified (of {$penguins->nrows()} rows, some incomplete)\n";

// plotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
// two-predictor model (bill measurements alone) just for visualization.
$billOnly = new NaiveBayesClassifier($penguins, 'species ~ bill_length_mm + bill_depth_mm');
echo "\nbill-measurements-only model training accuracy: " . cs(100 * $billOnly->trainingAccuracy()) . "%\n";
$billOnly->plotClassification($penguins, 'bill_length_mm', 'bill_depth_mm')->save('naive_bayes_penguins_classification.svg');
$billOnly->plotDecisionRegions('bill_length_mm', 'bill_depth_mm')->save('naive_bayes_penguins_decision_regions.svg');
echo "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n";
