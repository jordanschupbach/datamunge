<?php
// CMD: just TARGET=gaussian_process_regression_iris_ex run-php
// Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, GaussianProcessRegression};
use function Datamunge\Data\iris;

const FORMULA = 'Petal.Length ~ Petal.Width';

$iris = iris();
echo "iris: {$iris->nrows()} rows x {$iris->ncols()} cols\n";
echo 'formula: ' . FORMULA . "\n\n";

// Both the length scale and the noise ratio are auto-selected by maximizing the exact log
// marginal likelihood.
$model = new GaussianProcessRegression($iris, FORMULA);
$model->printSummary();

$model->plotFit($iris)->save('gpr_iris_fit.svg');
$model->plotLengthScaleProfile()->save('gpr_iris_length_scale_profile.svg');
echo "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n";

// Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
// interval at every point -- including far outside the training data, where it should widen
// substantially as the model's uncertainty grows.
$query = new DataFrame();
$query->addNumericColumn('Petal.Width', [0.2, 1.3, 2.5, 10.0]);
$detail = $model->predictFrame($query, 'confidence');
echo "\nPredictions with 95% confidence intervals:\n";
echo $detail, "\n";
echo "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n";
