<?php
// CMD: just TARGET=lda_iris_ex run-php
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Lda};
use function Datamunge\Data\iris;

$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n\n";

$model = new Lda($iris, 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width');
$model->printSummary();

echo "\nConfusion matrix (rows = actual, cols = predicted):\n{$model->confusionMatrix()}\n";

$newdata = new DataFrame();
$newdata->addNumericColumn('Sepal.Length', [5.1, 6.0, 6.5, 6.2]);
$newdata->addNumericColumn('Sepal.Width', [3.5, 2.7, 3.0, 2.8]);
$newdata->addNumericColumn('Petal.Length', [1.4, 4.5, 5.5, 4.8]);
$newdata->addNumericColumn('Petal.Width', [0.2, 1.5, 2.0, 1.8]);
echo "\nPredictions for new flowers:\n{$model->predictFrame($newdata)}\n";

$model->saveDiscriminantPlot('lda_iris_discriminants.svg');
echo "\nSaved discriminant plot as lda_iris_discriminants.svg\n";
