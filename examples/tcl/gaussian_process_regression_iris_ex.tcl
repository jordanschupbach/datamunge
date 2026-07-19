package require Datamunge 0.0.1

set FORMULA "Petal.Length ~ Petal.Width"

set iris [datamunge::DataFrame_iris]
puts "iris: [datamunge::DataFrame_nrows $iris] rows x [datamunge::DataFrame_ncols $iris] cols"
puts "formula: $FORMULA\n"

set model [datamunge::new_GaussianProcessRegression $iris $FORMULA]
datamunge::GaussianProcessRegression_print_summary $model

datamunge::Plot_save [datamunge::GaussianProcessRegression_plot_fit $model $iris] "gpr_iris_fit.svg"
datamunge::Plot_save [datamunge::GaussianProcessRegression_plot_length_scale_profile $model] "gpr_iris_length_scale_profile.svg"
puts "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg"

set query [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $query "Petal.Width" {0.2 1.3 2.5 10.0}
set detail [datamunge::GaussianProcessRegression_predict_frame $model $query "confidence"]
puts "\nPredictions with 95% confidence intervals:"
puts "[datamunge::DataFrame_to_string $detail]"
puts "(Petal.Width=10.0 is far outside the training range \[0.1, 2.5\] -- note how much wider its\n interval is than the in-range predictions.)"
