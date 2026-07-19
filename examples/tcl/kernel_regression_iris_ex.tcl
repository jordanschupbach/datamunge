package require Datamunge 0.0.1

set FORMULA "Petal.Length ~ Petal.Width"

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols"
puts "formula: $FORMULA\n"

set model [datamunge::new_KernelRegression $iris $FORMULA]
datamunge::KernelRegression_print_summary $model

datamunge::Plot_save [datamunge::KernelRegression_plot_fit $model $iris] "kernel_regression_iris_fit.svg"
datamunge::Plot_save [datamunge::KernelRegression_plot_cv_curve $model] "kernel_regression_iris_cv.svg"
puts "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg"

set small [datamunge::new_KernelRegression $iris $FORMULA "gaussian" 0.05]
puts "\nbandwidth=0.05 (too small): LOO R-squared=[datamunge::KernelRegression_r_squared $small]  LOO RMSE=[datamunge::KernelRegression_rmse $small]"
datamunge::Plot_save [datamunge::KernelRegression_plot_fit $small $iris] "kernel_regression_iris_fit_small_bandwidth.svg"

set large [datamunge::new_KernelRegression $iris $FORMULA "gaussian" 5.0]
puts "bandwidth=5.0 (too large):  LOO R-squared=[datamunge::KernelRegression_r_squared $large]  LOO RMSE=[datamunge::KernelRegression_rmse $large]"
datamunge::Plot_save [datamunge::KernelRegression_plot_fit $large $iris] "kernel_regression_iris_fit_large_bandwidth.svg"

puts "bandwidth=[datamunge::KernelRegression_bandwidth $model] (CV-selected): LOO R-squared=[datamunge::KernelRegression_r_squared $model]  LOO RMSE=[datamunge::KernelRegression_rmse $model]"
puts "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg"
