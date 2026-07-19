package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.KernelRegression;

public class KernelRegressionIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    String formula = "Petal.Length ~ Petal.Width";

    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols");
    System.out.println("formula: " + formula + "\n");

    var model = new KernelRegression(iris, formula);
    model.print_summary();

    model.plot_fit(iris).save("kernel_regression_iris_fit.svg");
    model.plot_cv_curve().save("kernel_regression_iris_cv.svg");
    System.out.println("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

    var small = new KernelRegression(iris, formula, "gaussian", 0.05);
    System.out.println("\nbandwidth=0.05 (too small): LOO R-squared=" + small.r_squared() + "  LOO RMSE=" + small.rmse());
    small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg");

    var large = new KernelRegression(iris, formula, "gaussian", 5.0);
    System.out.println("bandwidth=5.0 (too large):  LOO R-squared=" + large.r_squared() + "  LOO RMSE=" + large.rmse());
    large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg");

    System.out.println("bandwidth=" + model.bandwidth() + " (CV-selected): LOO R-squared=" + model.r_squared() + "  LOO RMSE=" + model.rmse());
    System.out.println("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
  }
}
