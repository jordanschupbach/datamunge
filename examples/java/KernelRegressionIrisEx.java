// CMD: just TARGET=kernel_regression_iris_ex run-java
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
import datamunge.KernelRegression;
import datamunge.data.Data;

public class KernelRegressionIrisEx {
  public static void main(String[] args) {
    String formula = "Petal.Length ~ Petal.Width";

    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols");
    System.out.println("formula: " + formula + "\n");

    // Bandwidth auto-selected via leave-one-out CV.
    var model = new KernelRegression(iris, formula);
    model.printSummary();
    System.out.flush();

    model.plotFit(iris).save("kernel_regression_iris_fit.svg");
    model.plotCvCurve().save("kernel_regression_iris_cv.svg");
    System.out.println("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

    // A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
    // bias-variance story. new KernelRegression(data, formula, kernel, bandwidth, ...)
    var small = new KernelRegression(iris, formula, "gaussian", 0.05);
    System.out.println(
        "\nbandwidth=0.05 (too small): LOO R-squared=" + small.rSquared()
            + "  LOO RMSE=" + small.rmse());
    small.plotFit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg");

    var large = new KernelRegression(iris, formula, "gaussian", 5.0);
    System.out.println(
        "bandwidth=5.0 (too large):  LOO R-squared=" + large.rSquared()
            + "  LOO RMSE=" + large.rmse());
    large.plotFit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg");

    System.out.println(
        "bandwidth=" + model.bandwidth() + " (CV-selected): LOO R-squared=" + model.rSquared()
            + "  LOO RMSE=" + model.rmse());
    System.out.println(
        "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and"
            + " kernel_regression_iris_fit_large_bandwidth.svg");
  }
}
