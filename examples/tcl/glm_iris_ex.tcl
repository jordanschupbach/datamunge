# CMD: just TARGET=glm_iris_ex run-tcl
# Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
package require datamunge

proc column {df name rows} {
    lmap i $rows {$df numeric_at $name $i}
}

proc mean {values} {
    expr {[tcl::mathop::+ {*}$values] / double([llength $values])}
}

set iris [datamunge::Data::iris]
set rows {}
for {set i 0} {$i < [$iris nrows]} {incr i} {
    lappend rows $i
}

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
set kept [lmap i $rows {
    if {[$iris string_at Species $i] ni {versicolor virginica}} continue
    set i
}]
set is_virginica [lmap i $kept {expr {[$iris string_at Species $i] eq "virginica" ? 1.0 : 0.0}}]
set petal_length [column $iris Petal.Length $kept]
set petal_width [column $iris Petal.Width $kept]

set sub [datamunge::DataFrame new]
$sub add_numeric_column Petal.Length $petal_length
$sub add_numeric_column Petal.Width $petal_width
$sub add_numeric_column is_virginica $is_virginica

puts "=================== Logistic regression (binomial, logit link) ==================="
set logit [datamunge::GLM new $sub "is_virginica ~ Petal.Length + Petal.Width" binomial]
$logit print_summary

set n [llength $is_virginica]
set correct 0
foreach f [$logit fitted_values] y $is_virginica {
    if {($f >= 0.5) == ($y >= 0.5)} {
        incr correct
    }
}
puts "\nResubstitution accuracy at 0.5 threshold: [expr {100.0 * $correct / $n}]%"

$logit save_diagnostic_plots glm_logistic_iris
puts "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg"

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
set grid_n 100
set width_mean [mean $petal_width]
set pl_min [expr {[tcl::mathfunc::min {*}$petal_length] - 0.3}]
set pl_max [expr {[tcl::mathfunc::max {*}$petal_length] + 0.3}]
set grid_x {}
for {set i 0} {$i < $grid_n} {incr i} {
    lappend grid_x [expr {$pl_min + ($pl_max - $pl_min) * $i / ($grid_n - 1)}]
}
set grid [datamunge::DataFrame new]
$grid add_numeric_column Petal.Length $grid_x
$grid add_numeric_column Petal.Width [lrepeat $grid_n $width_mean]
set curve_frame [$logit predict_frame $grid confidence]
puts "\nPredicted-probability curve (first 5 rows):"
puts [$curve_frame to_string 5]

# Poisson regression, for contrast: same IRLS engine, different family/link.
puts "\n=================== Poisson regression (log link) ==================="
set count_data [datamunge::DataFrame new]
$count_data add_numeric_column Sepal.Width [column $iris Sepal.Width $rows]
$count_data add_numeric_column Petal.Length [column $iris Petal.Length $rows]
$count_data add_numeric_column count \
    [lmap i $rows {expr {floor([$iris numeric_at Sepal.Length $i] + 0.5)}}]
set poisson [datamunge::GLM new $count_data "count ~ Sepal.Width + Petal.Length" poisson]
$poisson print_summary
