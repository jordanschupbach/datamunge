module app;

import std.stdio : writeln;
import datamunge;

void main() {
  string FORMULA = "Petal.Length ~ Petal.Width";

  auto iris = DataFrame.iris();
  writeln("iris: ", iris.nrows(), " rows x ", iris.ncols(), " cols");
  writeln("formula: ", FORMULA, "\n");

  auto model = new KernelRegression(iris, FORMULA);  // bandwidth auto-selected via leave-one-out CV
  model.print_summary();

  model.plot_fit(iris).save("kernel_regression_iris_fit.svg");
  model.plot_cv_curve().save("kernel_regression_iris_cv.svg");
  writeln("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

  auto small = new KernelRegression(iris, FORMULA, "gaussian", 0.05);
  writeln("\nbandwidth=0.05 (too small): LOO R-squared=", small.r_squared(), "  LOO RMSE=", small.rmse());
  small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg");

  auto large = new KernelRegression(iris, FORMULA, "gaussian", 5.0);
  writeln("bandwidth=5.0 (too large):  LOO R-squared=", large.r_squared(), "  LOO RMSE=", large.rmse());
  large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg");

  writeln("bandwidth=", model.bandwidth(), " (CV-selected): LOO R-squared=", model.r_squared(), "  LOO RMSE=", model.rmse());
  writeln("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
}
