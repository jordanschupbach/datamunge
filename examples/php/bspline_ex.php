<?php
// CMD: just TARGET=bspline_ex run-php
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Lm};

// Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
$x = $y = $z = [];
for ($iy = 0; $iy < 8; $iy++) {
    for ($ix = 0; $ix < 8; $ix++) {
        $xv = $ix / 7.0;
        $yv = $iy / 7.0;
        $x[] = $xv;
        $y[] = $yv;
        $z[] = sin($xv) + $yv * $yv;
    }
}
$data = new DataFrame();
$data->addNumericColumn('x', $x);
$data->addNumericColumn('y', $y);
$data->addNumericColumn('z', $z);

$surface = new Lm($data, 'z ~ bs(x, y)');
echo "Fitted z ~ bs(x, y)\n";
$surface->printSummary();

$newPoints = new DataFrame();
$newPoints->addNumericColumn('x', [0.25, 0.75]);
$newPoints->addNumericColumn('y', [0.50, 0.25]);
$pred = $surface->predict($newPoints);
printf("Predictions: [%.6f, %.6f]\n", $pred[0], $pred[1]);
