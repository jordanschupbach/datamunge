<?php

$FORMULA = "Petal.Length ~ Petal.Width";

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n");
print("formula: $FORMULA\n\n");

$model = new KernelRegression($iris, $FORMULA);
$model->print_summary();

$model->plot_fit($iris)->save("kernel_regression_iris_fit.svg");
$model->plot_cv_curve()->save("kernel_regression_iris_cv.svg");
print("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n");

$small = new KernelRegression($iris, $FORMULA, "gaussian", 0.05);
print("\nbandwidth=0.05 (too small): LOO R-squared=" . $small->r_squared() . "  LOO RMSE=" . $small->rmse() . "\n");
$small->plot_fit($iris)->save("kernel_regression_iris_fit_small_bandwidth.svg");

$large = new KernelRegression($iris, $FORMULA, "gaussian", 5.0);
print("bandwidth=5.0 (too large):  LOO R-squared=" . $large->r_squared() . "  LOO RMSE=" . $large->rmse() . "\n");
$large->plot_fit($iris)->save("kernel_regression_iris_fit_large_bandwidth.svg");

print("bandwidth=" . $model->bandwidth() . " (CV-selected): LOO R-squared=" . $model->r_squared() . "  LOO RMSE=" . $model->rmse() . "\n");
print("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n");

?>
