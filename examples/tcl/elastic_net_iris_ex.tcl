# CMD: just TARGET=elastic_net_iris_ex run-tcl
# Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
package require datamunge

set formula "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

set iris [datamunge::Data::iris]
puts "iris: [$iris nrows] rows x [$iris ncols] cols"
puts "formula: $formula\n"

# summary returns the same text print_summary writes, but goes through Tcl's stdout channel,
# so it stays in order with puts when stdout is redirected.
puts "=================== Ridge ==================="
set ridge [datamunge::Ridge new $iris $formula]
$ridge print_summary

puts "\n=================== Lasso ==================="
set lasso [datamunge::Lasso new $iris $formula]
$lasso print_summary

puts "\n================= Elastic Net ================="
# datamunge::ElasticNet new data formula ?alpha? ?lambda? ?n_lambda? ?cv_folds? ?standardize? ?seed?
set elastic [datamunge::ElasticNet new $iris $formula 0.5]
$elastic print_summary

puts "\nSaved figures showing how each model's coefficients respond to the regularization strength,\
      and the cross-validation curve used to pick it:"

foreach {name model label} [list \
        ridge $ridge "ridge:      " \
        lasso $lasso "lasso:      " \
        elasticnet $elastic "elastic net:"] {
    [$model plot_coefficient_path] save elastic_net_${name}_path.svg
    [$model plot_cv_curve] save elastic_net_${name}_cv.svg
    puts "  $label elastic_net_${name}_path.svg, elastic_net_${name}_cv.svg"
}
