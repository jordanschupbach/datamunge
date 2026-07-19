package require Datamunge 0.0.1

proc rosenbrock_dual {x0 x1} {
  set a [datamunge::Dual_subtract [datamunge::new_Dual 1.0 0.0] $x0]
  set b [datamunge::Dual_subtract $x1 [datamunge::Dual_multiply $x0 $x0]]
  return [datamunge::Dual_add [datamunge::Dual_multiply $a $a] [datamunge::Dual_multiply_scalar [datamunge::Dual_multiply $b $b] 100.0]]
}

proc rosenbrock_value {x0 x1} {
  set a [expr {1.0 - $x0}]
  set b [expr {$x1 - $x0 * $x0}]
  return [expr {$a * $a + 100.0 * $b * $b}]
}

proc rosenbrock_hyperdual {x0 x1} {
  set a [datamunge::HyperDual_subtract [datamunge::new_HyperDual 1.0 0.0 0.0 0.0] $x0]
  set b [datamunge::HyperDual_subtract $x1 [datamunge::HyperDual_multiply $x0 $x0]]
  return [datamunge::HyperDual_add [datamunge::HyperDual_multiply $a $a] [datamunge::HyperDual_multiply_scalar [datamunge::HyperDual_multiply $b $b] 100.0]]
}

puts "=================== Forward mode: scalar derivative ==================="
set x0 [datamunge::new_Dual 2.0 1.0]
set f [datamunge::Dual_subtract [datamunge::Dual_multiply [datamunge::Dual_multiply $x0 $x0] $x0] [datamunge::Dual_multiply_scalar $x0 2.0]]
puts "f(x) = x^3 - 2x, f'(2) = [datamunge::Dual_derivative $f] (exact: 10)"

puts "\n=================== Reverse mode: build a graph by hand ==================="
set tape [datamunge::new_Tape]
set a [datamunge::new_Var $tape 2.0]
set b [datamunge::new_Var $tape 3.0]
set y [datamunge::Var_add [datamunge::Var_multiply $a $b] [datamunge::Var_sin $a]]
puts "y = a*b + sin(a) at a=2, b=3 -> y = [datamunge::Var_value $y]"
set adjoint [datamunge::Tape_backward $tape $y]
puts "dy/da = [lindex $adjoint [datamunge::Var_index $a]] (exact: b + cos(a))"
puts "dy/db = [lindex $adjoint [datamunge::Var_index $b]] (exact: a)"

puts "\n=================== Forward vs reverse mode agree on the Rosenbrock function ==================="
set px 0.0
set py 0.0

set gx [datamunge::Dual_derivative [rosenbrock_dual [datamunge::new_Dual $px 1.0] [datamunge::new_Dual $py 0.0]]]
set gy [datamunge::Dual_derivative [rosenbrock_dual [datamunge::new_Dual $px 0.0] [datamunge::new_Dual $py 1.0]]]

set tape2 [datamunge::new_Tape]
set vx [datamunge::new_Var $tape2 $px]
set vy [datamunge::new_Var $tape2 $py]
set va [datamunge::Var_subtract [datamunge::new_Var $tape2 1.0] $vx]
set vb [datamunge::Var_subtract $vy [datamunge::Var_multiply $vx $vx]]
set vf [datamunge::Var_add [datamunge::Var_multiply $va $va] [datamunge::Var_multiply_scalar [datamunge::Var_multiply $vb $vb] 100.0]]
set grad_rev [datamunge::Tape_backward $tape2 $vf]

puts "f(0,0) = [rosenbrock_value $px $py]"
puts "gradient (forward mode): \[$gx, $gy\]"
puts "gradient (reverse mode): \[[lindex $grad_rev [datamunge::Var_index $vx]], [lindex $grad_rev [datamunge::Var_index $vy]]\]"

puts "\n=================== Jacobian of a vector-valued function ==================="
set vx_val 2.0
set vy_val 3.0

array set jac {}
set cols {{0 1.0 0.0} {1 0.0 1.0}}
foreach c $cols {
  set col [lindex $c 0]
  set seed_x [lindex $c 1]
  set seed_y [lindex $c 2]
  set ox [datamunge::Dual_multiply [datamunge::new_Dual $vx_val $seed_x] [datamunge::new_Dual $vx_val $seed_x]]
  set oy [datamunge::Dual_multiply [datamunge::new_Dual $vx_val $seed_x] [datamunge::new_Dual $vy_val $seed_y]]
  set oz [datamunge::Dual_multiply [datamunge::Dual_multiply [datamunge::new_Dual $vy_val $seed_y] [datamunge::new_Dual $vy_val $seed_y]] [datamunge::new_Dual $vy_val $seed_y]]
  set jac(0,$col) [datamunge::Dual_derivative $ox]
  set jac(1,$col) [datamunge::Dual_derivative $oy]
  set jac(2,$col) [datamunge::Dual_derivative $oz]
}
puts "f(x,y) = \[x^2, xy, y^3\] at (2,3), Jacobian:"
for {set row 0} {$row < 3} {incr row} {
  puts "  \[$jac($row,0), $jac($row,1)\]"
}

puts "\n=================== Hessian via second-order forward mode (HyperDual) ==================="
set mx 1.0
set my 1.0

set seeds {{1.0 0.0} {0.0 1.0}}
array set h {}
for {set i 0} {$i < 2} {incr i} {
  for {set j 0} {$j < 2} {incr j} {
    set e1 [lindex $seeds $i]
    set e2 [lindex $seeds $j]
    set hx [datamunge::new_HyperDual $mx [lindex $e1 0] [lindex $e2 0] 0.0]
    set hy [datamunge::new_HyperDual $my [lindex $e1 1] [lindex $e2 1] 0.0]
    set h($i,$j) [datamunge::HyperDual_eps1eps2 [rosenbrock_hyperdual $hx $hy]]
  }
}
puts "Hessian of the Rosenbrock function at its minimum (1,1):"
for {set row 0} {$row < 2} {incr row} {
  puts "  \[$h($row,0), $h($row,1)\]"
}

puts "\n=================== Gradient descent driven by reverse-mode gradients ==================="
set point_x -1.2
set point_y 1.0
set learning_rate 0.001
set n_steps 2000
for {set step 0} {$step < $n_steps} {incr step} {
  set t [datamunge::new_Tape]
  set vx2 [datamunge::new_Var $t $point_x]
  set vy2 [datamunge::new_Var $t $point_y]
  set va2 [datamunge::Var_subtract [datamunge::new_Var $t 1.0] $vx2]
  set vb2 [datamunge::Var_subtract $vy2 [datamunge::Var_multiply $vx2 $vx2]]
  set vf2 [datamunge::Var_add [datamunge::Var_multiply $va2 $va2] [datamunge::Var_multiply_scalar [datamunge::Var_multiply $vb2 $vb2] 100.0]]
  set grad [datamunge::Tape_backward $t $vf2]
  set loss [datamunge::Var_value $vf2]
  set point_x [expr {$point_x - $learning_rate * [lindex $grad [datamunge::Var_index $vx2]]}]
  set point_y [expr {$point_y - $learning_rate * [lindex $grad [datamunge::Var_index $vy2]]}]
  if {$step == 0 || $step == $n_steps - 1} {
    puts "step $step: loss = $loss, x = \[$point_x, $point_y\]"
  }
}
puts "(true minimum is at \[1, 1\] with loss 0)"
