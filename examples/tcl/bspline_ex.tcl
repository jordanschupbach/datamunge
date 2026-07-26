package require Datamunge 0.0.1

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

# Plain Tcl lists auto-convert to the DVector parameters of add_numeric_column (scalar fast-path).
set data [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $data "x" $x
datamunge::DataFrame_add_numeric_column $data "y" $y
datamunge::DataFrame_add_numeric_column $data "z" $z

set surface [datamunge::new_LM $data "z ~ bs(x, y)"]
puts "Fitted z ~ bs(x, y)"
datamunge::LM_print_summary $surface

set new_points [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $new_points "x" {0.25 0.75}
datamunge::DataFrame_add_numeric_column $new_points "y" {0.50 0.25}
# LM_predict returns a std::vector<double>, which the binding converts to a plain Tcl list.
set pred [datamunge::LM_predict $surface $new_points]
puts [format "Predictions: \[%.6f, %.6f\]" [lindex $pred 0] [lindex $pred 1]]
