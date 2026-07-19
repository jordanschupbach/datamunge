<?php

// Note: SWIG's PHP backend renames the "Var" class to "c_Var" since "var" is a reserved PHP
// keyword (confirmed via the "'var' is a PHP keyword, renaming to 'c_Var'" SWIG warning).

function rosenbrock_dual($x0, $x1) {
    $a = (new Dual(1.0, 0.0))->subtract($x0);
    $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

function rosenbrock_value($x0, $x1) {
    $a = 1.0 - $x0;
    $b = $x1 - $x0 * $x0;
    return $a * $a + 100.0 * $b * $b;
}

function rosenbrock_hyperdual($x0, $x1) {
    $a = (new HyperDual(1.0, 0.0, 0.0, 0.0))->subtract($x0);
    $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

print("=================== Forward mode: scalar derivative ===================\n");
$x0 = new Dual(2.0, 1.0);
$f = $x0->multiply($x0)->multiply($x0)->subtract($x0->multiply_scalar(2.0));
print("f(x) = x^3 - 2x, f'(2) = " . $f->derivative() . " (exact: 10)\n");

print("\n=================== Reverse mode: build a graph by hand ===================\n");
$tape = new Tape();
$a = new c_Var($tape, 2.0);
$b = new c_Var($tape, 3.0);
$y = $a->multiply($b)->add($a->sin());
print("y = a*b + sin(a) at a=2, b=3 -> y = " . $y->value() . "\n");
$adjoint = $tape->backward($y);
print("dy/da = " . $adjoint->get($a->index()) . " (exact: b + cos(a))\n");
print("dy/db = " . $adjoint->get($b->index()) . " (exact: a)\n");

print("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n");
$px = 0.0;
$py = 0.0;

$gx = rosenbrock_dual(new Dual($px, 1.0), new Dual($py, 0.0))->derivative();
$gy = rosenbrock_dual(new Dual($px, 0.0), new Dual($py, 1.0))->derivative();

$tape2 = new Tape();
$vx = new c_Var($tape2, $px);
$vy = new c_Var($tape2, $py);
$va = (new c_Var($tape2, 1.0))->subtract($vx);
$vb = $vy->subtract($vx->multiply($vx));
$vf = $va->multiply($va)->add($vb->multiply($vb)->multiply_scalar(100.0));
$grad_rev = $tape2->backward($vf);

print("f(0,0) = " . rosenbrock_value($px, $py) . "\n");
print("gradient (forward mode): [$gx, $gy]\n");
print("gradient (reverse mode): [" . $grad_rev->get($vx->index()) . ", " . $grad_rev->get($vy->index()) . "]\n");

print("\n=================== Jacobian of a vector-valued function ===================\n");
$vx_val = 2.0;
$vy_val = 3.0;

$jac = array(array(0, 0), array(0, 0), array(0, 0));
foreach (array(array(0, 1.0, 0.0), array(1, 0.0, 1.0)) as $c) {
    list($col, $seed_x, $seed_y) = $c;
    $ox = (new Dual($vx_val, $seed_x))->multiply(new Dual($vx_val, $seed_x));
    $oy = (new Dual($vx_val, $seed_x))->multiply(new Dual($vy_val, $seed_y));
    $oz = (new Dual($vy_val, $seed_y))->multiply(new Dual($vy_val, $seed_y))->multiply(new Dual($vy_val, $seed_y));
    $jac[0][$col] = $ox->derivative();
    $jac[1][$col] = $oy->derivative();
    $jac[2][$col] = $oz->derivative();
}
print("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n");
for ($row = 0; $row < 3; $row++) {
    print("  [" . $jac[$row][0] . ", " . $jac[$row][1] . "]\n");
}

print("\n=================== Hessian via second-order forward mode (HyperDual) ===================\n");
$mx = 1.0;
$my = 1.0;

$seeds = array(array(1.0, 0.0), array(0.0, 1.0));
$h = array(array(0, 0), array(0, 0));
for ($i = 0; $i < 2; $i++) {
    for ($j = 0; $j < 2; $j++) {
        $e1 = $seeds[$i];
        $e2 = $seeds[$j];
        $hx = new HyperDual($mx, $e1[0], $e2[0], 0.0);
        $hy = new HyperDual($my, $e1[1], $e2[1], 0.0);
        $h[$i][$j] = rosenbrock_hyperdual($hx, $hy)->eps1eps2();
    }
}
print("Hessian of the Rosenbrock function at its minimum (1,1):\n");
for ($row = 0; $row < 2; $row++) {
    print("  [" . $h[$row][0] . ", " . $h[$row][1] . "]\n");
}

print("\n=================== Gradient descent driven by reverse-mode gradients ===================\n");
$point_x = -1.2;
$point_y = 1.0;
$learning_rate = 0.001;
$n_steps = 2000;
for ($step = 0; $step < $n_steps; $step++) {
    $t = new Tape();
    $vx2 = new c_Var($t, $point_x);
    $vy2 = new c_Var($t, $point_y);
    $va2 = (new c_Var($t, 1.0))->subtract($vx2);
    $vb2 = $vy2->subtract($vx2->multiply($vx2));
    $vf2 = $va2->multiply($va2)->add($vb2->multiply($vb2)->multiply_scalar(100.0));
    $grad = $t->backward($vf2);
    $loss = $vf2->value();
    $point_x -= $learning_rate * $grad->get($vx2->index());
    $point_y -= $learning_rate * $grad->get($vy2->index());
    if ($step === 0 || $step === $n_steps - 1) {
        print("step $step: loss = $loss, x = [$point_x, $point_y]\n");
    }
}
print("(true minimum is at [1, 1] with loss 0)\n");

?>
