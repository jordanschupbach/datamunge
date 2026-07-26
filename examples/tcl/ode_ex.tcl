package require Datamunge 0.0.1

# NOTE: the Python/Ruby ode examples also demonstrate a live, user-supplied RHS by subclassing
# datamunge.RHS (a SWIG director). Stock SWIG's Tcl backend generates no director code, so RHS
# cannot be subclassed from Tcl; this example uses the built-in named systems via solve_builtin().

# state_at()/solve_builtin's vector returns come back as plain Tcl lists (the binding's
# specialize_std_vector out-typemaps), so these use list ops rather than DVector_get.
proc last_state {sol} {
  return [datamunge::ODESolution_state_at $sol [expr {[datamunge::ODESolution_size $sol] - 1}]]
}

proc state_to_list {st} {
  set out {}
  foreach x $st { lappend out [format %g $x] }
  return $out
}

puts "=================== Every named built-in system (no director needed) ==================="
set solver [datamunge::new_ODESolver]
# {system {params...} {y0...}}
set systems {
  {exponential_decay {1.0} {1.0}}
  {logistic_growth {1.0 1.0} {0.5}}
  {harmonic_oscillator {1.0} {1.0 0.0}}
  {van_der_pol {1.0} {2.0 0.0}}
  {lorenz {10.0 28.0 2.6666666667} {1.0 1.0 1.0}}
}
foreach s $systems {
  set name [lindex $s 0]
  set params [lindex $s 1]
  set y0 [lindex $s 2]
  set sol [datamunge::ODESolver_solve_builtin $solver $name $params $y0 0.0 1.0]
  set parts [state_to_list [last_state $sol]]
  puts [format "%-22s steps=%-6d final state=\[%s\]" $name \
    [datamunge::ODESolution_steps_taken_get $sol] [join $parts ", "]]
}

puts "\n=================== Harmonic oscillator (energy conservation) ==================="
set options [datamunge::new_ODEOptions]
datamunge::ODEOptions_method_set $options $::StepMethod_RK4
datamunge::ODEOptions_step_size_set $options 0.01
set solver [datamunge::new_ODESolver $options]
set sol [datamunge::ODESolver_solve_builtin $solver "harmonic_oscillator" {1.0} {1.0 0.0} 0.0 20.0]
set final [last_state $sol]
set x [lindex $final 0]
set v [lindex $final 1]
puts [format "x(20) = %.6f (cos(20) = %.6f)" $x [expr {cos(20)}]]
puts [format "energy x^2+v^2 = %.6f (should stay near 1.0)" [expr {$x * $x + $v * $v}]]

set t_series {}
set x_series {}
set v_series {}
for {set i 0} {$i < [datamunge::ODESolution_size $sol]} {incr i} {
  lappend t_series [datamunge::ODESolution_time_at $sol $i]
  set st [datamunge::ODESolution_state_at $sol $i]
  lappend x_series [lindex $st 0]
  lappend v_series [lindex $st 1]
}
set plot [datamunge::RPlot_plot $t_series $x_series "l" "x(t)"]
datamunge::RPlot_lines $plot $t_series $v_series "v(t)"
datamunge::Plot_title $plot "Harmonic Oscillator"
datamunge::Plot_x_label $plot "t"
datamunge::Plot_y_label $plot "state"
datamunge::Plot_save_svg $plot "ode_harmonic_oscillator_tcl.svg"
puts "wrote ode_harmonic_oscillator_tcl.svg"

puts "\n=================== Lorenz attractor (phase plane) ==================="
set options [datamunge::new_ODEOptions]
datamunge::ODEOptions_method_set $options $::StepMethod_RK4
datamunge::ODEOptions_step_size_set $options 0.005
set solver [datamunge::new_ODESolver $options]
set sol [datamunge::ODESolver_solve_builtin $solver "lorenz" {10.0 28.0 2.6666666667} {1.0 1.0 1.0} 0.0 25.0]
puts [format "steps_taken = %d" [datamunge::ODESolution_steps_taken_get $sol]]

set xs {}
set zs {}
for {set i 0} {$i < [datamunge::ODESolution_size $sol]} {incr i} {
  set st [datamunge::ODESolution_state_at $sol $i]
  lappend xs [lindex $st 0]
  lappend zs [lindex $st 2]
}
set plot [datamunge::RPlot_plot $xs $zs "l" "trajectory"]
datamunge::Plot_title $plot "Lorenz Attractor (x-z phase plane)"
datamunge::Plot_x_label $plot "x"
datamunge::Plot_y_label $plot "z"
datamunge::Plot_save_svg $plot "ode_lorenz_phase_plane_tcl.svg"
puts "wrote ode_lorenz_phase_plane_tcl.svg"
