# CMD: just TARGET=optim_advanced_ex run-tcl
# Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Tcl.
package require datamunge

# Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
# EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
# ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
# implement the objective. From Tcl, subclass the interface class with TclOO and define its
# virtual methods; the optimizer then calls the Tcl methods. The coordinates are passed as a
# variable name and updated in place.

# f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
oo::class create Quadratic {
    superclass ::datamunge::HessianFunction
    method evaluate {c} {
        lassign $c x y
        expr {($x - 1)**2 + 10 * ($y + 2)**2}
    }
    method gradient {c} {
        lassign $c x y
        list [expr {2 * ($x - 1)}] [expr {20 * ($y + 2)}]
    }
    method hessian {c} {
        return {{2.0 0.0} {0.0 20.0}}
    }
}

# Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
oo::class create Exponential {
    superclass ::datamunge::ResidualFunction
    variable ts
    constructor {times} {
        next
        set ts $times
    }
    method residuals {p} {
        lassign $p a b
        lmap t $ts {expr {$a * exp($b * $t) - 2 * exp(0.5 * $t)}}
    }
    method jacobian {p} {
        lassign $p a b
        lmap t $ts {list [expr {exp($b * $t)}] [expr {$a * $t * exp($b * $t)}]}
    }
}

# Leaves gradient and hessian out.
oo::class create Incomplete {
    superclass ::datamunge::HessianFunction
    method evaluate {c} {
        return 0.0
    }
}

puts "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ==================="
# Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
# then check the acquisition function favors an unexplored region.
set xs {0.0 0.5 1.0 1.5 2.0 3.0}
set ys [lmap x $xs {expr {($x - 1)**2}}]
set surrogate [datamunge::RBFGaussianProcessSurrogate new 1.0 1e-6]
$surrogate fit [lmap x $xs {list $x}] $ys
set incumbent [tcl::mathfunc::min 0.0 {*}$ys]  ;# best observed value so far
puts "incumbent (best observed y): $incumbent"
foreach x {0.25 1.0 2.5} {
    puts [format "acquisition(x=%.2f) = %.6f" $x [$surrogate acquisition [list $x] $incumbent]]
}

puts "\n=================== Options structs + optimizer construction (all 10 new classes) ==================="
# Every new Options struct and optimizer class constructs cleanly, proving
# the binding for the whole batch is wired correctly.
# A new datamunge Options object of the given class with some fields set.
proc options {class args} {
    set options [datamunge::$class new]
    if {[llength $args]} {
        $options configure {*}$args
    }
    return $options
}
set pg [datamunge::ProximalGradient new [options ProximalGradientOptions -step_size 0.05]]
set fista [datamunge::FISTA new [options FISTAOptions]]
set lm [datamunge::LevenbergMarquardt new [options LevenbergMarquardtOptions -max_iterations 200]]
set newton [datamunge::Newton new [options NewtonOptions]]
set trn [datamunge::TrustRegionNewton new [options TrustRegionNewtonOptions]]
set al [datamunge::AugmentedLagrangian new [options AugmentedLagrangianOptions]]
set sqp [datamunge::SQP new [options SQPOptions]]
set ip [datamunge::InteriorPoint new [options InteriorPointOptions]]
set bo [datamunge::BayesianOptimization new [options BayesianOptimizationOptions -initial_samples 8]]
foreach optimizer [list $pg $fista $lm $newton $trn $al $sqp $ip $bo] {
    set name [namespace tail [info object class $optimizer]]
    puts "$name: ${name}()"
}

puts "\n=================== Tcl objectives (subclassing the C++ interfaces) ==================="
# Newton's method on the quadratic bowl.
set start {5.0 5.0}
set value [$newton optimize [Quadratic new] start]
puts [format "Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))" $value {*}$start]

# Levenberg-Marquardt fitting y = a * exp(b t).
set params {1.0 0.1}
set sse [$lm optimize [Exponential new {0.0 0.5 1.0 1.5 2.0}] params]
puts [format "LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)" $sse {*}$params]

# An interface method a subclass leaves out raises when the optimizer calls it.
set coordinates {0.0 0.0}
set ok 1
set mentions 0
try {
    $newton optimize [Incomplete new] coordinates
} trap {DATAMUNGE} {msg} {
    set ok 0
    set mentions [string match *gradient* $msg]
}
puts "missing gradient(): ok=[expr {$ok ? "true" : "false"}], error mentions it: [expr {$mentions ? "true" : "false"}]"
