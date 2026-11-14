# CMD: just TARGET=kernel_regression_iris_ex run-tcl
# Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
package require datamunge

set formula "Petal.Length ~ Petal.Width"

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols"
puts "formula: $formula\n"

# Bandwidth auto-selected via leave-one-out CV.
set model [datamunge::KernelRegression new $iris $formula]
$model print_summary

[$model plot_fit $iris] save kernel_regression_iris_fit.svg
[$model plot_cv_curve] save kernel_regression_iris_cv.svg
puts "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg"

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story. KernelRegression new data formula ?kernel? ?bandwidth? ...
set small [datamunge::KernelRegression new $iris $formula gaussian 0.05]
puts "\nbandwidth=0.05 (too small): LOO R-squared=[$small r_squared]  LOO RMSE=[$small rmse]"
[$small plot_fit $iris] save kernel_regression_iris_fit_small_bandwidth.svg

set large [datamunge::KernelRegression new $iris $formula gaussian 5.0]
puts "bandwidth=5.0 (too large):  LOO R-squared=[$large r_squared]  LOO RMSE=[$large rmse]"
[$large plot_fit $iris] save kernel_regression_iris_fit_large_bandwidth.svg

puts "bandwidth=[$model bandwidth] (CV-selected): LOO R-squared=[$model r_squared]  LOO RMSE=[$model rmse]"
puts "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg"
