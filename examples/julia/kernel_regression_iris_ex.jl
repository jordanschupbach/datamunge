# CMD: just TARGET=kernel_regression_iris_ex run-julia

# Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
using Datamunge

const formula = "Petal.Length ~ Petal.Width"

const iris = Datamunge.Data.iris()
println("iris: $(nrows(iris)) rows x $(ncols(iris)) cols")
println("formula: $formula\n")

const model = KernelRegression(iris, formula)  # bandwidth auto-selected via leave-one-out CV
print_summary(model)

save(plot_fit(model, iris), "kernel_regression_iris_fit.svg")
save(plot_cv_curve(model), "kernel_regression_iris_cv.svg")
println("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg")

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story. KernelRegression(data, formula, kernel, bandwidth, n_bandwidth, standardize)
let small = KernelRegression(iris, formula, "gaussian", 0.05)
    println("\nbandwidth=0.05 (too small): LOO R-squared=$(r_squared(small))  LOO RMSE=$(rmse(small))")
    save(plot_fit(small, iris), "kernel_regression_iris_fit_small_bandwidth.svg")
end

let large = KernelRegression(iris, formula, "gaussian", 5.0)
    println("bandwidth=5.0 (too large):  LOO R-squared=$(r_squared(large))  LOO RMSE=$(rmse(large))")
    save(plot_fit(large, iris), "kernel_regression_iris_fit_large_bandwidth.svg")
end

println("bandwidth=$(bandwidth(model)) (CV-selected): LOO R-squared=$(r_squared(model))  LOO RMSE=$(rmse(model))")
println("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg")
