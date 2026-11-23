<?php
// CMD: just TARGET=clustering_ex run-php
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{AgglomerativeClustering, Dbscan, KMeans};
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
$species = array_map(fn(int $i): string => $iris->stringAt('Species', $i), range(0, $iris->nrows() - 1));
$featureColumns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];

/**
 * For each cluster, print its size and its majority species (negative labels are noise).
 *
 * @param list<int> $clusterLabels
 * @param list<string> $species
 */
function reportPurity(array $clusterLabels, int $nClusters, array $species): void
{
    $votes = array_fill(0, $nClusters, []);
    foreach (array_map(null, $clusterLabels, $species) as [$label, $sp]) {
        if ($label >= 0) {
            $votes[$label][$sp] = ($votes[$label][$sp] ?? 0) + 1;
        }
    }
    for ($c = 0; $c < $nClusters; $c++) {
        $total = array_sum($votes[$c]);
        if ($total === 0) {
            continue;
        }
        // The first species with the most votes, as C#'s MaxBy picks it.
        $best = max($votes[$c]);
        $bestSpecies = array_search($best, $votes[$c], true);
        echo "  cluster $c: $total points, majority $bestSpecies ($best/$total)\n";
    }
}

echo "=================== K-means (k=3) on iris ===================\n";
$kmeans = new KMeans($iris, $featureColumns, 3);
echo 'Inertia: ' . cs($kmeans->inertia()) . ", iterations (best run): {$kmeans->iterationsUsed()}\n";
reportPurity($kmeans->labels(), 3, $species);

echo "\n=================== Agglomerative clustering on iris ===================\n";
foreach (['ward', 'complete', 'average'] as $linkage) {
    $model = new AgglomerativeClustering($iris, $featureColumns, 3, $linkage);
    echo "--- linkage=$linkage ---\n";
    reportPurity($model->labels(), 3, $species);
}

echo "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n";
$wardModel = new AgglomerativeClustering($iris, $featureColumns, 3, 'ward');
reportPurity($wardModel->cut(2), 2, $species);

echo "\n=================== DBSCAN on iris ===================\n";
$dbscan = new Dbscan($iris, $featureColumns, 0.6, 5);
echo "Clusters found: {$dbscan->nClusters()}, noise points: {$dbscan->nNoise()} (of {$dbscan->observations()})\n";
reportPurity($dbscan->labels(), $dbscan->nClusters(), $species);
