from pydatamunge import datamunge as dm

FORMULA = "Petal.Length ~ Petal.Width"

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols")
print(f"formula: {FORMULA}\n")

model = dm.KernelRegression(iris, FORMULA)  # bandwidth auto-selected via leave-one-out CV
model.print_summary()

model.plot_fit(iris).save("kernel_regression_iris_fit.svg")
model.plot_cv_curve().save("kernel_regression_iris_cv.svg")
print("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg")

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story. KernelRegression(data, formula, kernel, bandwidth, n_bandwidth, standardize)
small = dm.KernelRegression(iris, FORMULA, "gaussian", 0.05)
print(f"\nbandwidth=0.05 (too small): LOO R-squared={small.r_squared()}  LOO RMSE={small.rmse()}")
small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg")

large = dm.KernelRegression(iris, FORMULA, "gaussian", 5.0)
print(f"bandwidth=5.0 (too large):  LOO R-squared={large.r_squared()}  LOO RMSE={large.rmse()}")
large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg")

print(f"bandwidth={model.bandwidth()} (CV-selected): LOO R-squared={model.r_squared()}  LOO RMSE={model.rmse()}")
print("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and "
      "kernel_regression_iris_fit_large_bandwidth.svg")
