<?php
// CMD: just TARGET=elastic_net_iris_ex run-php
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{ElasticNet, Lasso, Ridge};
use function Datamunge\Data\iris;

const FORMULA = 'Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width';

$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n";
echo 'formula: ' . FORMULA . "\n\n";

echo "=================== Ridge ===================\n";
$ridge = new Ridge($iris, FORMULA);
$ridge->printSummary();

echo "\n=================== Lasso ===================\n";
$lasso = new Lasso($iris, FORMULA);
$lasso->printSummary();

echo "\n================= Elastic Net =================\n";
// new ElasticNet(data, formula, alpha=0.5, lambda, nLambda, cvFolds, standardize, seed)
$elastic = new ElasticNet($iris, FORMULA, 0.5);
$elastic->printSummary();

echo "\nSaved figures showing how each model's coefficients respond to the regularization strength, "
    . "and the cross-validation curve used to pick it:\n";

$figures = [
    [$ridge->plotCoefficientPath(), $ridge->plotCvCurve(), 'ridge', 'ridge:      '],
    [$lasso->plotCoefficientPath(), $lasso->plotCvCurve(), 'lasso', 'lasso:      '],
    [$elastic->plotCoefficientPath(), $elastic->plotCvCurve(), 'elasticnet', 'elastic net:'],
];
foreach ($figures as [$path, $cv, $name, $label]) {
    $path->save("elastic_net_{$name}_path.svg");
    $cv->save("elastic_net_{$name}_cv.svg");
    echo "  $label elastic_net_{$name}_path.svg, elastic_net_{$name}_cv.svg\n";
}
