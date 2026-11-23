<?php
// CMD: just TARGET=mds_ex run-php
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\Mds;
use function Datamunge\Data\iris;

$iris = iris();
$featureColumns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];
$species = array_map(fn(int $i): string => $iris->stringAt('Species', $i), range(0, $iris->nrows() - 1));

echo "=================== Classical MDS on iris (euclidean) ===================\n";
$mds = new Mds($iris, $featureColumns, 2, 'euclidean');
$mds->printSummary();

echo "\nEmbedding as a DataFrame:\n{$mds->embeddingFrame()}\n";

$mds->plotEmbeddingGrouped($species)->saveSvg('mds_iris_embedding.svg');
echo "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n";

echo "\n=================== Classical MDS on iris (manhattan) ===================\n";
$manhattanMds = new Mds($iris, $featureColumns, 2, 'manhattan');
printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n", $mds->goodnessOfFit(), $manhattanMds->goodnessOfFit());
