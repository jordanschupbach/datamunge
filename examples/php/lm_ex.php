<?php
// CMD: just TARGET=lm_ex run-php
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Lm};

$cars = new DataFrame();
$cars->addNumericColumn('hp', [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]);
$cars->addNumericColumn('wt', [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]);
$cars->addStringColumn('transmission', ['manual', 'manual', 'manual', 'automatic', 'automatic',
                                        'automatic', 'automatic', 'automatic', 'automatic', 'automatic']);
$cars->addNumericColumn('mpg', [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]);

echo "Fitting: mpg ~ hp + wt + transmission\n\n";
$model = new Lm($cars, 'mpg ~ hp + wt + transmission');
$model->printSummary();

echo "\nSequential ANOVA:\n{$model->anova()}\n";

$newcars = new DataFrame();
$newcars->addNumericColumn('hp', [150.0, 90.0]);
$newcars->addNumericColumn('wt', [3.0, 2.5]);
$newcars->addStringColumn('transmission', ['manual', 'automatic']);
echo "\nPredictions with 95% confidence intervals:\n{$model->predictFrame($newcars, 'confidence')}\n";

$model->saveDiagnosticPlots('lm_ex_diagnostics');
echo "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n";
