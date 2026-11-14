# CMD: just TARGET=autodiff_ex run-tcl
# Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
package require datamunge
namespace import datamunge::Dual datamunge::Tape datamunge::Var datamunge::HyperDual

# Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
# (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
# bound. This port demonstrates the same ideas with the bound named-method API.
# [$var index] is the Var's position in the adjoint list [$tape backward $output] returns.

proc rosenbrock_dual {x0 x1} {
    set a [[Dual new 1.0 0.0] subtract $x0]
    set b [$x1 subtract [$x0 multiply $x0]]
    return [[$a multiply $a] add [[$b multiply $b] multiply_scalar 100.0]]
}

proc rosenbrock_value {x0 x1} {
    expr {(1.0 - $x0)**2 + 100.0 * ($x1 - $x0 * $x0)**2}
}

# Build the Rosenbrock graph on tape; returns the output Var.
proc rosenbrock_var {tape x0 x1} {
    set a [[Var new $tape 1.0] subtract $x0]
    set b [$x1 subtract [$x0 multiply $x0]]
    return [[$a multiply $a] add [[$b multiply $b] multiply_scalar 100.0]]
}

proc rosenbrock_hyper_dual {x0 x1} {
    set a [[HyperDual new 1.0 0.0 0.0 0.0] subtract $x0]
    set b [$x1 subtract [$x0 multiply $x0]]
    return [[$a multiply $a] add [[$b multiply $b] multiply_scalar 100.0]]
}

proc adjoint {adjoints v} {
    lindex $adjoints [$v index]
}

puts "=================== Forward mode: scalar derivative ==================="
set x0 [Dual new 2.0 1.0] ;# seed derivative = 1 to read df/dx directly
set f [[[$x0 multiply $x0] multiply $x0] subtract [$x0 multiply_scalar 2.0]] ;# x^3 - 2x
puts "f(x) = x^3 - 2x, f'(2) = [$f derivative] (exact: 10)"

puts "\n=================== Reverse mode: build a graph by hand ==================="
set tape [Tape new]
set a [Var new $tape 2.0]
set b [Var new $tape 3.0]
set y [[$a multiply $b] add [$a sin]]
puts "y = a*b + sin(a) at a=2, b=3 -> y = [$y value]"
set adj [$tape backward $y]
puts "dy/da = [adjoint $adj $a] (exact: b + cos(a))"
puts "dy/db = [adjoint $adj $b] (exact: a)"

puts "\n=================== Forward vs reverse mode agree on the Rosenbrock function ==================="
set px 0.0
set py 0.0
# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
set gx [[rosenbrock_dual [Dual new $px 1.0] [Dual new $py 0.0]] derivative]
set gy [[rosenbrock_dual [Dual new $px 0.0] [Dual new $py 1.0]] derivative]
# Reverse mode: one Tape/Var pass computes every partial at once.
set tape [Tape new]
set vx [Var new $tape $px]
set vy [Var new $tape $py]
set grad [$tape backward [rosenbrock_var $tape $vx $vy]]
puts "f(0,0) = [rosenbrock_value $px $py]"
puts "gradient (forward mode): \[$gx, $gy\]"
puts "gradient (reverse mode): \[[adjoint $grad $vx], [adjoint $grad $vy]\]"

puts "\n=================== Jacobian of a vector-valued function ==================="
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
proc vector_fn {x y} {
    list [$x multiply $x] [$x multiply $y] [[$y multiply $y] multiply $y]
}
set jac [lrepeat 3 [lrepeat 2 {}]]
foreach {col sx sy} {0 1.0 0.0  1 0.0 1.0} {
    set row 0
    foreach out [vector_fn [Dual new 2.0 $sx] [Dual new 3.0 $sy]] {
        lset jac $row $col [$out derivative]
        incr row
    }
}
puts "f(x,y) = \[x^2, xy, y^3\] at (2,3), Jacobian:"
foreach r $jac {
    puts "  \[[join $r {, }]\]"
}

puts "\n=================== Hessian via second-order forward mode (HyperDual) ==================="
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
set mx 1.0
set my 1.0 ;# the Rosenbrock function's minimum
set seeds {{1.0 0.0} {0.0 1.0}}
set h [lmap e1 $seeds {
    lmap e2 $seeds {
        [rosenbrock_hyper_dual \
            [HyperDual new $mx [lindex $e1 0] [lindex $e2 0] 0.0] \
            [HyperDual new $my [lindex $e1 1] [lindex $e2 1] 0.0]] eps1eps2
    }
}]
puts "Hessian of the Rosenbrock function at its minimum (1,1):"
foreach r $h {
    puts "  \[[join $r {, }]\]"
}

puts "\n=================== Gradient descent driven by reverse-mode gradients ==================="
set point {-1.2 1.0} ;# the classic Rosenbrock start
set learning_rate 0.001
set n_steps 2000
for {set step 0} {$step < $n_steps} {incr step} {
    set tape [Tape new]
    set vx [Var new $tape [lindex $point 0]]
    set vy [Var new $tape [lindex $point 1]]
    set vf [rosenbrock_var $tape $vx $vy]
    set grad [$tape backward $vf]
    set loss [$vf value]
    lset point 0 [expr {[lindex $point 0] - $learning_rate * [adjoint $grad $vx]}]
    lset point 1 [expr {[lindex $point 1] - $learning_rate * [adjoint $grad $vy]}]
    if {$step == 0 || $step == $n_steps - 1} {
        puts "step $step: loss = $loss, x = \[[join $point {, }]\]"
    }
}
puts "(true minimum is at \[1, 1\] with loss 0)"
