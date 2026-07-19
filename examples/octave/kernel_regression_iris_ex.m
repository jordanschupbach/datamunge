1;

datamunge;

FORMULA = "Petal.Length ~ Petal.Width";

iris = DataFrame_iris();
printf("iris: %d rows x %d cols\n", DataFrame_nrows(iris), DataFrame_ncols(iris));
printf("formula: %s\n\n", FORMULA);

model = KernelRegression(iris, FORMULA);  % bandwidth auto-selected via leave-one-out CV
KernelRegression_print_summary(model);

Plot_save(KernelRegression_plot_fit(model, iris), "kernel_regression_iris_fit.svg");
Plot_save(KernelRegression_plot_cv_curve(model), "kernel_regression_iris_cv.svg");
printf("\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n");

% A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
% bias-variance story. KernelRegression(data, formula, kernel, bandwidth, n_bandwidth, standardize)
small = KernelRegression(iris, FORMULA, "gaussian", 0.05);
printf("\nbandwidth=0.05 (too small): LOO R-squared=%g  LOO RMSE=%g\n", KernelRegression_r_squared(small), KernelRegression_rmse(small));
Plot_save(KernelRegression_plot_fit(small, iris), "kernel_regression_iris_fit_small_bandwidth.svg");

large = KernelRegression(iris, FORMULA, "gaussian", 5.0);
printf("bandwidth=5.0 (too large):  LOO R-squared=%g  LOO RMSE=%g\n", KernelRegression_r_squared(large), KernelRegression_rmse(large));
Plot_save(KernelRegression_plot_fit(large, iris), "kernel_regression_iris_fit_large_bandwidth.svg");

printf("bandwidth=%g (CV-selected): LOO R-squared=%g  LOO RMSE=%g\n", KernelRegression_bandwidth(model), KernelRegression_r_squared(model), KernelRegression_rmse(model));
printf("\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n");
