using System;

class Program {
  static void Main() {
    const string FORMULA = "Petal.Length ~ Petal.Width";

    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols");
    Console.WriteLine($"formula: {FORMULA}\n");

    var model = new KernelRegression(iris, FORMULA);
    model.print_summary();

    model.plot_fit(iris).save("kernel_regression_iris_fit.svg");
    model.plot_cv_curve().save("kernel_regression_iris_cv.svg");
    Console.WriteLine("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

    var small = new KernelRegression(iris, FORMULA, "gaussian", 0.05);
    Console.WriteLine($"\nbandwidth=0.05 (too small): LOO R-squared={small.r_squared()}  LOO RMSE={small.rmse()}");
    small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg");

    var large = new KernelRegression(iris, FORMULA, "gaussian", 5.0);
    Console.WriteLine($"bandwidth=5.0 (too large):  LOO R-squared={large.r_squared()}  LOO RMSE={large.rmse()}");
    large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg");

    Console.WriteLine($"bandwidth={model.bandwidth()} (CV-selected): LOO R-squared={model.r_squared()}  LOO RMSE={model.rmse()}");
    Console.WriteLine("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
  }
}
