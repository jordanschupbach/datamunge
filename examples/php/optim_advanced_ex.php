<?php

// Exercises the datamunge::optim module in PHP -- previously the ENTIRE module (including
// the original ArbitraryFunction/DifferentiableFunction base interfaces) was %ignore'd for
// PHP because SWIG director support had never been turned on (%module had no
// directors="1"). This example proves both the base of the module (a plain
// DifferentiableFunction subclass through GradientDescent) and three of the ten newest
// optimizer classes work end-to-end through real PHP-side director callbacks.

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function to_array($vec) {
    $n = $vec->size();
    $out = array();
    for ($i = 0; $i < $n; $i++) $out[] = $vec->get($i);
    return $out;
}

function fmt($values, $precision = 6) {
    $parts = array();
    foreach ($values as $v) $parts[] = number_format($v, $precision);
    return "[" . implode(", ", $parts) . "]";
}

// ==================================================================================
// (a) Base-of-module sanity check: a plain DifferentiableFunction subclass run through
//     GradientDescent. This never worked in PHP before this change (the whole optim
//     module, base interfaces included, was ignored).
// ==================================================================================
class QuadraticBowl extends DifferentiableFunction {
    private $target;
    function __construct($target) {
        parent::__construct();
        $this->target = $target;
    }
    function evaluate($coordinates) {
        $sum = 0.0;
        $n = $coordinates->size();
        for ($i = 0; $i < $n; $i++) {
            $d = $coordinates->get($i) - $this->target[$i];
            $sum += $d * $d;
        }
        return $sum;
    }
    function gradient($coordinates) {
        $n = $coordinates->size();
        $g = new DVector($n);
        for ($i = 0; $i < $n; $i++) {
            $g->set($i, 2.0 * ($coordinates->get($i) - $this->target[$i]));
        }
        return $g;
    }
}

print("=================== DifferentiableFunction director -> GradientDescent ===================\n");
$target = array(3.0, -2.0, 1.5);
$f = new QuadraticBowl($target);
$x = dvector(array(0.0, 0.0, 0.0));
$opts = new GradientDescentOptions();
$opts->step_size = 0.1;
$opts->momentum = 0.0;
$opts->max_iterations = 2000;
$opts->tolerance = 1e-12;
$value = (new GradientDescent($opts))->optimize($f, $x);
print("GradientDescent: f=" . $value . " x=" . fmt(to_array($x)) . " (target=" . fmt($target) . ")\n");

// ==================================================================================
// (b) HessianFunction director -> Newton, one of the 10 newest classes. A simple 2-D
//     quadratic bowl with an exact, constant Hessian -- Newton should land almost exactly
//     on the minimum in very few iterations.
// ==================================================================================
class HessianBowl extends HessianFunction {
    private $target;
    function __construct($target) {
        parent::__construct();
        $this->target = $target;
    }
    function evaluate($coordinates) {
        $dx = $coordinates->get(0) - $this->target[0];
        $dy = $coordinates->get(1) - $this->target[1];
        return $dx * $dx + $dy * $dy;
    }
    function gradient($coordinates) {
        $g = new DVector(2);
        $g->set(0, 2.0 * ($coordinates->get(0) - $this->target[0]));
        $g->set(1, 2.0 * ($coordinates->get(1) - $this->target[1]));
        return $g;
    }
    function hessian($coordinates) {
        $row0 = dvector(array(2.0, 0.0));
        $row1 = dvector(array(0.0, 2.0));
        $hv = new DVectorVector(0);
        $hv->push($row0);
        $hv->push($row1);
        return $hv;
    }
}

print("\n=================== HessianFunction director -> Newton (new class) ===================\n");
$htarget = array(5.0, -4.0);
$hf = new HessianBowl($htarget);
$hx = dvector(array(10.0, 10.0));
$hvalue = (new Newton())->optimize($hf, $hx);
print("Newton: f=" . $hvalue . " x=" . fmt(to_array($hx)) . " (target=" . fmt($htarget) . ")\n");

// ==================================================================================
// (c) ResidualFunction director -> LevenbergMarquardt (new class), fitting
//     y = A*exp(-k*t) + c to synthetic noise-free data generated from known parameters,
//     confirming the recovered parameters match the ground truth.
// ==================================================================================
class ExpDecayResiduals extends ResidualFunction {
    private $t;
    private $y;
    function __construct($t, $y) {
        parent::__construct();
        $this->t = $t;
        $this->y = $y;
    }
    function residuals($params) {
        $A = $params->get(0);
        $k = $params->get(1);
        $c = $params->get(2);
        $n = count($this->t);
        $r = new DVector($n);
        for ($i = 0; $i < $n; $i++) {
            $pred = $A * exp(-$k * $this->t[$i]) + $c;
            $r->set($i, $pred - $this->y[$i]);
        }
        return $r;
    }
    function jacobian($params) {
        $A = $params->get(0);
        $k = $params->get(1);
        $n = count($this->t);
        $jv = new DVectorVector(0);
        for ($i = 0; $i < $n; $i++) {
            $t_i = $this->t[$i];
            $e = exp(-$k * $t_i);
            $row = dvector(array($e, -$A * $t_i * $e, 1.0));
            $jv->push($row);
        }
        return $jv;
    }
}

print("\n=================== ResidualFunction director -> LevenbergMarquardt (new class) ===================\n");
$true_A = 5.0;
$true_k = 0.5;
$true_c = 1.0;
$t_data = array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
$y_data = array();
foreach ($t_data as $t_i) $y_data[] = $true_A * exp(-$true_k * $t_i) + $true_c;

$lm_f = new ExpDecayResiduals($t_data, $y_data);
$params = dvector(array(1.0, 1.0, 0.0));
$lm_value = (new LevenbergMarquardt())->optimize($lm_f, $params);
print("LevenbergMarquardt: sum-sq-residuals/2=" . $lm_value
    . " params=" . fmt(to_array($params))
    . " (true=" . fmt(array($true_A, $true_k, $true_c)) . ")\n");

// ==================================================================================
// (d) BayesianOptimization (new class) with the ready-to-use RBFGaussianProcessSurrogate
//     -- no BayesianSurrogate subclass needed, only the ArbitraryFunction objective.
// ==================================================================================
class Simple1D extends ArbitraryFunction {
    function evaluate($coordinates) {
        $d = $coordinates->get(0) - 2.0;
        return $d * $d;
    }
}

print("\n=================== BayesianOptimization + RBFGaussianProcessSurrogate (new class) ===================\n");
$bo_f = new Simple1D();
$bo_x = dvector(array(0.0));
$lower = dvector(array(-5.0));
$upper = dvector(array(5.0));
$surrogate = new RBFGaussianProcessSurrogate();
$bo_value = (new BayesianOptimization())->optimize($bo_f, $bo_x, $lower, $upper, $surrogate);
print("BayesianOptimization: f=" . $bo_value . " x=" . fmt(to_array($bo_x)) . " (target=[2.000000])\n");
