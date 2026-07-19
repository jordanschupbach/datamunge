package require Datamunge 0.0.1

set iris [datamunge::DataFrame_iris]

set is_virginica {}
set petal_length {}
set petal_width {}
set n [datamunge::DataFrame_nrows $iris]
for {set i 0} {$i < $n} {incr i} {
  set species [datamunge::DataFrame_string_at $iris "Species" $i]
  if {$species eq "versicolor" || $species eq "virginica"} {
    lappend is_virginica [expr {$species eq "virginica" ? 1.0 : 0.0}]
    lappend petal_length [datamunge::DataFrame_numeric_at $iris "Petal.Length" $i]
    lappend petal_width [datamunge::DataFrame_numeric_at $iris "Petal.Width" $i]
  }
}

set sub [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $sub "Petal.Length" $petal_length
datamunge::DataFrame_add_numeric_column $sub "Petal.Width" $petal_width
datamunge::DataFrame_add_numeric_column $sub "is_virginica" $is_virginica

puts "=================== Logistic regression (binomial, logit link) ==================="
set logit [datamunge::new_GLM $sub "is_virginica ~ Petal.Length + Petal.Width" "binomial"]
datamunge::GLM_print_summary $logit

set fitted [datamunge::GLM_fitted_values $logit]
set correct 0
set nn [llength $is_virginica]
for {set i 0} {$i < $nn} {incr i} {
  set fv [lindex $fitted $i]
  set av [lindex $is_virginica $i]
  if {($fv >= 0.5) == ($av >= 0.5)} {
    incr correct
  }
}
puts "\nResubstitution accuracy at 0.5 threshold: [expr {100.0 * $correct / $nn}]%"

datamunge::GLM_save_diagnostic_plots $logit "glm_logistic_iris"
puts "\nSaved glm_logistic_iris_\{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage\}.svg"

set width_mean [expr {[tcl::mathop::+ {*}$petal_width] / double($nn)}]
set grid_n 100
set pl_min [expr {[tcl::mathfunc::min {*}$petal_length] - 0.3}]
set pl_max [expr {[tcl::mathfunc::max {*}$petal_length] + 0.3}]
set grid_x {}
set grid_w {}
for {set i 0} {$i < $grid_n} {incr i} {
  lappend grid_x [expr {$pl_min + ($pl_max - $pl_min) * $i / double($grid_n - 1)}]
  lappend grid_w $width_mean
}
set grid [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $grid "Petal.Length" $grid_x
datamunge::DataFrame_add_numeric_column $grid "Petal.Width" $grid_w
set curve_frame [datamunge::GLM_predict_frame $logit $grid "confidence"]
puts "\nPredicted-probability curve (first 5 rows):"
puts "[datamunge::DataFrame_to_string $curve_frame 5]"

puts "\n=================== Poisson regression (log link) ==================="
set count {}
set sepal_width {}
set all_petal_length {}
for {set i 0} {$i < $n} {incr i} {
  lappend count [expr {round([datamunge::DataFrame_numeric_at $iris "Sepal.Length" $i])}]
  lappend sepal_width [datamunge::DataFrame_numeric_at $iris "Sepal.Width" $i]
  lappend all_petal_length [datamunge::DataFrame_numeric_at $iris "Petal.Length" $i]
}
set count_data [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $count_data "Sepal.Width" $sepal_width
datamunge::DataFrame_add_numeric_column $count_data "Petal.Length" $all_petal_length
datamunge::DataFrame_add_numeric_column $count_data "count" $count

set poisson [datamunge::new_GLM $count_data "count ~ Sepal.Width + Petal.Length" "poisson"]
datamunge::GLM_print_summary $poisson
