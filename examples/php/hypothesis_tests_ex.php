<?php
// CMD: just TARGET=hypothesis_tests_ex run-php
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{Alternative, DataFrame, HypothesisTestResult};
use function Datamunge\{binomial_test, chi_squared_test_independence, f_test_variance, fisher_exact_test_2x2,
    ks_test_one_sample_normal, kruskal_wallis_test, one_way_anova, pearson_correlation_test, proportion_test_one_sample,
    shapiro_francia_test, spearman_correlation_test, t_test_two_sample, wilcoxon_rank_sum_test};
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

function printResult(string $label, HypothesisTestResult $r): void
{
    $alternative = match ($r->alternative) {
        Alternative::Less => 'less',
        Alternative::Greater => 'greater',
        default => 'two.sided',
    };
    echo "$label: statistic=" . cs($r->statistic);
    if ($r->parameter1 != 0) {
        echo ', df1=' . cs($r->parameter1);
    }
    if ($r->parameter2 != 0) {
        echo ', df2=' . cs($r->parameter2);
    }
    echo ', p=' . cs($r->pValue) . " ($alternative)";
    if ($r->hasConfInt) {
        echo ', CI=[' . cs($r->confIntLower) . ', ' . cs($r->confIntUpper) . ']';
    }
    echo " -- {$r->method}\n";
}

/** @return list<float> */
function columnForSpecies(DataFrame $df, string $column, string $species): array
{
    $values = [];
    for ($i = 0; $i < $df->nrows(); $i++) {
        if ($df->stringAt('Species', $i) === $species) {
            $values[] = $df->numericAt($column, $i);
        }
    }
    return $values;
}

/** @return list<float> */
function column(DataFrame $df, string $column): array
{
    $values = [];
    for ($i = 0; $i < $df->nrows(); $i++) {
        $values[] = $df->numericAt($column, $i);
    }
    return $values;
}

$iris = iris();

$setosaPetal = columnForSpecies($iris, 'Petal.Length', 'setosa');
$versicolorPetal = columnForSpecies($iris, 'Petal.Length', 'versicolor');
$virginicaPetal = columnForSpecies($iris, 'Petal.Length', 'virginica');

echo "=================== t-tests: petal length, setosa vs. versicolor ===================\n";
printResult('Welch two-sample t-test', t_test_two_sample($setosaPetal, $versicolorPetal));
printResult('Wilcoxon rank-sum test', wilcoxon_rank_sum_test($setosaPetal, $versicolorPetal));

echo "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n";
$allPetal = [...$setosaPetal, ...$versicolorPetal, ...$virginicaPetal];
$sizes = [count($setosaPetal), count($versicolorPetal), count($virginicaPetal)];
printResult('One-way ANOVA', one_way_anova($allPetal, $sizes));
printResult('Kruskal-Wallis', kruskal_wallis_test($allPetal, $sizes));

echo "\n=================== Correlation: sepal length vs. petal length ===================\n";
$sepalLength = column($iris, 'Sepal.Length');
$petalLength = column($iris, 'Petal.Length');
printResult('Pearson correlation', pearson_correlation_test($sepalLength, $petalLength));
printResult('Spearman correlation', spearman_correlation_test($sepalLength, $petalLength));

echo "\n=================== F-test: petal length variance, setosa vs. virginica ===================\n";
printResult('F test', f_test_variance($setosaPetal, $virginicaPetal));

echo "\n=================== Normality: is sepal length normally distributed within setosa? ===================\n";
$setosaSepal = columnForSpecies($iris, 'Sepal.Length', 'setosa');
printResult('Shapiro-Francia', shapiro_francia_test($setosaSepal));
printResult('KS vs. fitted normal', ks_test_one_sample_normal($setosaSepal, 5.006, 0.3525));

echo "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n";
$sorted = $allPetal;
sort($sorted);
$medianAll = $sorted[intdiv(count($allPetal), 2)];
$countLong = fn(array $v): int => count(array_filter($v, fn(float $x): bool => $x > $medianAll));
$setosaLong = $countLong($setosaPetal);
$setosaShort = count($setosaPetal) - $setosaLong;
$versicolorLong = $countLong($versicolorPetal);
$versicolorShort = count($versicolorPetal) - $versicolorLong;
echo "table: setosa=[$setosaLong,$setosaShort] versicolor=[$versicolorLong,$versicolorShort]\n";
$table = [(float) $setosaLong, (float) $setosaShort, (float) $versicolorLong, (float) $versicolorShort];
printResult('Chi-squared independence', chi_squared_test_independence($table, 2, 2));
printResult("Fisher's exact test", fisher_exact_test_2x2($setosaLong, $setosaShort, $versicolorLong, $versicolorShort));

echo "\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n";
$longCount = $countLong($allPetal);
$n = count($allPetal);
printResult('One-sample proportion test (vs 0.5)', proportion_test_one_sample($longCount, $n, 0.5));
printResult('Exact binomial test (vs 0.5)', binomial_test($longCount, $n, 0.5));
