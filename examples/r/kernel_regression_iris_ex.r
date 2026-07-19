# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
library(datamunger)

FORMULA <- "Petal.Length ~ Petal.Width"

iris <- DataFrame_iris()
cat(DataFrame_nrows(iris), "rows x", DataFrame_ncols(iris), "cols\n")
cat("formula:", FORMULA, "\n\n")

model <- KernelRegression(iris, FORMULA) # bandwidth auto-selected via leave-one-out CV
KernelRegression_print_summary(model)

Plot_save(KernelRegression_plot_fit(model, iris), "kernel_regression_iris_fit.svg")
Plot_save(KernelRegression_plot_cv_curve(model), "kernel_regression_iris_cv.svg")
cat("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n")

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story. KernelRegression(data, formula, kernel, bandwidth, n_bandwidth, standardize)
small <- KernelRegression(iris, FORMULA, "gaussian", 0.05)
cat("\nbandwidth=0.05 (too small): LOO R-squared=", KernelRegression_r_squared(small),
    "  LOO RMSE=", KernelRegression_rmse(small), "\n")
Plot_save(KernelRegression_plot_fit(small, iris), "kernel_regression_iris_fit_small_bandwidth.svg")

large <- KernelRegression(iris, FORMULA, "gaussian", 5.0)
cat("bandwidth=5.0 (too large):  LOO R-squared=", KernelRegression_r_squared(large),
    "  LOO RMSE=", KernelRegression_rmse(large), "\n")
Plot_save(KernelRegression_plot_fit(large, iris), "kernel_regression_iris_fit_large_bandwidth.svg")

cat("bandwidth=", KernelRegression_bandwidth(model), "(CV-selected): LOO R-squared=", KernelRegression_r_squared(model),
    "  LOO RMSE=", KernelRegression_rmse(model), "\n")
cat("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n")
