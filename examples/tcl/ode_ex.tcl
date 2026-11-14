# CMD: just TARGET=ode_ex run-tcl
# ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
package require datamunge

proc last_state {sol} {
    return [$sol state_at [expr {[$sol size] - 1}]]
}

# The time vector followed by one list per requested state component (0-based), one value per step.
proc solution_series {sol args} {
    set times {}
    set states {}
    for {set i 0} {$i < [$sol size]} {incr i} {
        lappend times [$sol time_at $i]
        lappend states [$sol state_at $i]
    }
    set series [list $times]
    foreach c $args {
        lappend series [lmap s $states {lindex $s $c}]
    }
    return $series
}

proc rk4_solver {step} {
    set options [datamunge::ODEOptions new]
    $options configure -method RK4 -step_size $step
    return [datamunge::ODESolver new $options]
}

# NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

puts "=================== Every named built-in system (no director needed) ==================="
set solver [datamunge::ODESolver new]
foreach {name params y0} [list \
    exponential_decay   {1.0}                             {1.0} \
    logistic_growth     {1.0 1.0}                         {0.5} \
    harmonic_oscillator {1.0}                             {1.0 0.0} \
    van_der_pol         {1.0}                             {2.0 0.0} \
    lorenz              [list 10.0 28.0 [expr {8.0 / 3}]] {1.0 1.0 1.0} \
] {
    set sol [$solver solve_builtin $name $params $y0 0.0 1.0]
    set state [join [lmap v [last_state $sol] {format %g $v}] {, }]
    puts [format "%-22s steps=%-6d final state=\[%s\]" $name [$sol steps_taken] $state]
}

puts "\n=================== Harmonic oscillator (energy conservation) ==================="
set sol [[rk4_solver 0.01] solve_builtin harmonic_oscillator {1.0} {1.0 0.0} 0.0 20.0]
lassign [last_state $sol] x v
puts [format "x(20) = %.6f (cos(20) = %.6f)" $x [expr {cos(20.0)}]]
puts [format "energy x^2+v^2 = %.6f (should stay near 1.0)" [expr {$x * $x + $v * $v}]]
lassign [solution_series $sol 0 1] ts xs vs
set p [datamunge::RPlot plot $ts $xs l x(t)]
$p lines $ts $vs v(t)
[[[$p title "Harmonic Oscillator"] x_label t] y_label state] save_svg ode_harmonic_oscillator_tcl.svg
puts "wrote ode_harmonic_oscillator_tcl.svg"

puts "\n=================== Lorenz attractor (phase plane) ==================="
set sol [[rk4_solver 0.005] solve_builtin lorenz [list 10.0 28.0 [expr {8.0 / 3}]] {1.0 1.0 1.0} 0.0 25.0]
puts "steps_taken = [$sol steps_taken]"
lassign [solution_series $sol 0 2] _ xs zs
set p [datamunge::RPlot plot $xs $zs l trajectory]
[[[$p title "Lorenz Attractor (x-z phase plane)"] x_label x] y_label z] save_svg ode_lorenz_phase_plane_tcl.svg
puts "wrote ode_lorenz_phase_plane_tcl.svg"
