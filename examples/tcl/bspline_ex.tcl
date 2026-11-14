# CMD: just TARGET=bspline_ex run-tcl
# Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
package require datamunge

# Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
set x {}
set y {}
set z {}
for {set iy 0} {$iy < 8} {incr iy} {
    for {set ix 0} {$ix < 8} {incr ix} {
        set xv [expr {$ix / 7.0}]
        set yv [expr {$iy / 7.0}]
        lappend x $xv
        lappend y $yv
        lappend z [expr {sin($xv) + $yv * $yv}]
    }
}
set data [datamunge::DataFrame new]
$data add_numeric_column x $x
$data add_numeric_column y $y
$data add_numeric_column z $z

set surface [datamunge::LM new $data "z ~ bs(x, y)"]
puts "Fitted z ~ bs(x, y)"
$surface print_summary

set new_points [datamunge::DataFrame new]
$new_points add_numeric_column x {0.25 0.75}
$new_points add_numeric_column y {0.50 0.25}
set pred [$surface predict $new_points]
puts [format "Predictions: \[%.6f, %.6f\]" {*}$pred]
