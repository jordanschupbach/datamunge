# CMD: just TARGET=gaussian_process_regression_iris_ex run-tcl
# Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
package require datamunge

set formula "Petal.Length ~ Petal.Width"

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols"
puts "formula: $formula\n"

# Both the length scale and the noise ratio are auto-selected by maximizing the exact log
# marginal likelihood.
set model [datamunge::GaussianProcessRegression new $iris $formula]
$model print_summary

[$model plot_fit $iris] save gpr_iris_fit.svg
[$model plot_length_scale_profile] save gpr_iris_length_scale_profile.svg
puts "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg"

# Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
# interval at every point -- including far outside the training data, where it should widen
# substantially as the model's uncertainty grows.
set query [datamunge::DataFrame new]
$query add_numeric_column Petal.Width {0.2 1.3 2.5 10.0}
set detail [$model predict_frame $query confidence]
puts "\nPredictions with 95% confidence intervals:"
puts [$detail to_string]
puts "(Petal.Width=10.0 is far outside the training range \[0.1, 2.5\] -- note how much wider its\n interval is than the in-range predictions.)"
