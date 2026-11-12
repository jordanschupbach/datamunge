# CMD: nix develop .#r --command Rscript examples/r/kernel_regression_iris_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
library(datamunger)

FORMULA <- "Petal.Length ~ Petal.Width"

iris <- DataFrame$iris()
cat(iris$nrows(), "rows x", iris$ncols(), "cols\n")
cat("formula:", FORMULA, "\n\n")

model <- KernelRegression$new(iris, FORMULA) # bandwidth auto-selected via leave-one-out CV
model$print_summary()

model$plot_fit(iris)$save("kernel_regression_iris_fit.svg")
model$plot_cv_curve()$save("kernel_regression_iris_cv.svg")
cat("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n")

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story. KernelRegression(data, formula, kernel, bandwidth, n_bandwidth, standardize)
small <- KernelRegression$new(iris, FORMULA, "gaussian", 0.05)
cat("\nbandwidth=0.05 (too small): LOO R-squared=", small$r_squared(),
    "  LOO RMSE=", small$rmse(), "\n")
small$plot_fit(iris)$save("kernel_regression_iris_fit_small_bandwidth.svg")

large <- KernelRegression$new(iris, FORMULA, "gaussian", 5.0)
cat("bandwidth=5.0 (too large):  LOO R-squared=", large$r_squared(),
    "  LOO RMSE=", large$rmse(), "\n")
large$plot_fit(iris)$save("kernel_regression_iris_fit_large_bandwidth.svg")

cat("bandwidth=", model$bandwidth(), "(CV-selected): LOO R-squared=", model$r_squared(),
    "  LOO RMSE=", model$rmse(), "\n")
cat("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n")
