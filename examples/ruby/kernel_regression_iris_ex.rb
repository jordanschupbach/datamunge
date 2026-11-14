# CMD: just TARGET=kernel_regression_iris_ex run-ruby
# Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
require "datamunge"

formula = "Petal.Length ~ Petal.Width"

iris = Datamunge::Data.iris
puts "iris: #{iris.nrows} rows x #{iris.ncols} cols"
puts "formula: #{formula}\n\n"

model = Datamunge::KernelRegression.new(iris, formula) # bandwidth auto-selected via leave-one-out CV
model.print_summary

model.plot_fit(iris).save("kernel_regression_iris_fit.svg")
model.plot_cv_curve.save("kernel_regression_iris_cv.svg")
puts "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg"

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story.
small = Datamunge::KernelRegression.new(iris, formula, bandwidth: 0.05)
puts "\nbandwidth=0.05 (too small): LOO R-squared=#{small.r_squared}  LOO RMSE=#{small.rmse}"
small.plot_fit(iris).save("kernel_regression_iris_fit_small_bandwidth.svg")

large = Datamunge::KernelRegression.new(iris, formula, bandwidth: 5.0)
puts "bandwidth=5.0 (too large):  LOO R-squared=#{large.r_squared}  LOO RMSE=#{large.rmse}"
large.plot_fit(iris).save("kernel_regression_iris_fit_large_bandwidth.svg")

puts "bandwidth=#{model.bandwidth} (CV-selected): LOO R-squared=#{model.r_squared}  LOO RMSE=#{model.rmse}"
puts "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg"
