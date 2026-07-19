const datamunge = require("../../index.js");

const FORMULA = "Petal.Length ~ Petal.Width";

const iris = datamunge.DataFrame.iris();
console.log(`iris: ${iris.nrows()} rows x ${iris.ncols()} cols`);
console.log(`formula: ${FORMULA}\n`);

const model = new datamunge.KernelRegression(iris, FORMULA);
model.print_summary();

model.plot_fit(iris).save("kernel_regression_iris_fit.svg");
model.plot_cv_curve().save("kernel_regression_iris_cv.svg");
console.log("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

const small = new datamunge.KernelRegression(iris, FORMULA, "gaussian", 0.05);
console.log(`\nbandwidth=0.05 (too small): LOO R-squared=${small.r_squared()}  LOO RMSE=${small.rmse()}`);
small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg");

const large = new datamunge.KernelRegression(iris, FORMULA, "gaussian", 5.0);
console.log(`bandwidth=5.0 (too large):  LOO R-squared=${large.r_squared()}  LOO RMSE=${large.rmse()}`);
large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg");

console.log(`bandwidth=${model.bandwidth()} (CV-selected): LOO R-squared=${model.r_squared()}  LOO RMSE=${model.rmse()}`);
console.log("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
