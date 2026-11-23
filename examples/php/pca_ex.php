<?php
// CMD: just TARGET=pca_ex run-php
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\Pca;
use function Datamunge\Data\iris;

$iris = iris();
$featureColumns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];

// species labels, for the grouped (colored-by-species) scores plot
$species = array_map(fn(int $i): string => $iris->stringAt('Species', $i), range(0, $iris->nrows() - 1));

echo "=================== PCA on iris (scaled) ===================\n";
$pca = new Pca($iris, $featureColumns, true, true);
$pca->printSummary();

echo "\nPC1 loadings (which original features drive it):\n";
foreach (array_map(null, $pca->featureNames(), $pca->componentLoadings(0)) as [$name, $loading]) {
    printf("  %s: %.4f\n", $name, $loading);
}

echo "\nTraining scores as a DataFrame:\n{$pca->scoresFrame()}\n";

$pca->plotScoresGrouped($species)->saveSvg('pca_iris_scores.svg');
echo "\nScores scatter (colored by species) saved as pca_iris_scores.svg\n";

$pca->plotScree()->saveSvg('pca_iris_scree.svg');
echo "Scree plot saved as pca_iris_scree.svg\n";

echo "\n=================== PCA on iris (unscaled) ===================\n";
$unscaled = new Pca($iris, $featureColumns, true, false);
printf("PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n",
    100 * $unscaled->explainedVarianceRatio()[0], 100 * $pca->explainedVarianceRatio()[0]);
