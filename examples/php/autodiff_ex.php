<?php
// CMD: just TARGET=autodiff_ex run-php
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{Dual, HyperDual, Tape, Var_};

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// Var_::index() is the Var's position in the adjoint vector Tape::backward returns. (The C++
// class Var is Var_ in PHP, where `var` is a reserved word.)

/** A double the way C# prints it (shortest round-trip, e.g. 6, 0.1, 1E-05). */
function cs(float $x): string
{
    if (is_nan($x)) {
        return 'NaN';
    }
    if (is_infinite($x)) {
        return $x > 0 ? '∞' : '-∞';
    }
    if ($x == 0.0) {
        return fdiv(1.0, $x) < 0 ? '-0' : '0';
    }
    for ($p = 0; $p < 17; $p++) {
        $s = sprintf("%.{$p}e", $x);
        if ((float) $s === $x) {
            break;
        }
    }
    [$mant, $exp] = explode('e', $s);
    $exp = (int) $exp;
    $sign = $mant[0] === '-' ? '-' : '';
    $digits = rtrim(str_replace(['-', '.'], '', $mant), '0');
    if ($exp < -5 || $exp >= 15) {
        $body = $digits[0] . (strlen($digits) > 1 ? '.' . substr($digits, 1) : '')
            . 'E' . ($exp < 0 ? '-' : '+') . sprintf('%02d', abs($exp));
    } elseif ($exp < 0) {
        $body = '0.' . str_repeat('0', -$exp - 1) . $digits;
    } elseif (strlen($digits) <= $exp + 1) {
        $body = $digits . str_repeat('0', $exp + 1 - strlen($digits));
    } else {
        $body = substr($digits, 0, $exp + 1) . '.' . substr($digits, $exp + 1);
    }
    return $sign . $body;
}

function rosenbrockDual(Dual $x0, Dual $x1): Dual
{
    $a = (new Dual(1.0, 0.0))->subtract($x0);
    $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiplyScalar(100.0));
}

function rosenbrockValue(float $x0, float $x1): float
{
    return (1.0 - $x0) ** 2 + 100.0 * ($x1 - $x0 * $x0) ** 2;
}

// Build the Rosenbrock graph on tape; returns the output Var_.
function rosenbrockVar(Tape $tape, Var_ $x0, Var_ $x1): Var_
{
    $a = (new Var_($tape, 1.0))->subtract($x0);
    $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiplyScalar(100.0));
}

function rosenbrockHyperDual(HyperDual $x0, HyperDual $x1): HyperDual
{
    $a = (new HyperDual(1.0, 0.0, 0.0, 0.0))->subtract($x0);
    $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiplyScalar(100.0));
}

/** @param list<float> $adjoints */
function adjoint(array $adjoints, Var_ $v): float
{
    return $adjoints[$v->index()];
}

echo "=================== Forward mode: scalar derivative ===================\n";
$x0 = new Dual(2.0, 1.0);  // seed derivative = 1 to read df/dx directly
$f = $x0->multiply($x0)->multiply($x0)->subtract($x0->multiplyScalar(2.0));  // x^3 - 2x
echo "f(x) = x^3 - 2x, f'(2) = " . cs($f->derivative()) . " (exact: 10)\n";

echo "\n=================== Reverse mode: build a graph by hand ===================\n";
$tape = new Tape();
$a = new Var_($tape, 2.0);
$b = new Var_($tape, 3.0);
$y = $a->multiply($b)->add($a->sin());
echo 'y = a*b + sin(a) at a=2, b=3 -> y = ' . cs($y->value()) . "\n";
$adjoints = $tape->backward($y);
echo 'dy/da = ' . cs(adjoint($adjoints, $a)) . " (exact: b + cos(a))\n";
echo 'dy/db = ' . cs(adjoint($adjoints, $b)) . " (exact: a)\n";

echo "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n";
[$px, $py] = [0.0, 0.0];
// Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
$gx = rosenbrockDual(new Dual($px, 1.0), new Dual($py, 0.0))->derivative();
$gy = rosenbrockDual(new Dual($px, 0.0), new Dual($py, 1.0))->derivative();
// Reverse mode: one Tape/Var_ pass computes every partial at once.
$tape = new Tape();
$vx = new Var_($tape, $px);
$vy = new Var_($tape, $py);
$grad = $tape->backward(rosenbrockVar($tape, $vx, $vy));
echo 'f(0,0) = ' . cs(rosenbrockValue($px, $py)) . "\n";
echo 'gradient (forward mode): [' . cs($gx) . ', ' . cs($gy) . "]\n";
echo 'gradient (reverse mode): [' . cs(adjoint($grad, $vx)) . ', ' . cs(adjoint($grad, $vy)) . "]\n";

echo "\n=================== Jacobian of a vector-valued function ===================\n";
// f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
$vectorFn = fn(Dual $x, Dual $y): array => [$x->multiply($x), $x->multiply($y), $y->multiply($y)->multiply($y)];
$jac = array_fill(0, 3, [0.0, 0.0]);
foreach ([[0, 1.0, 0.0], [1, 0.0, 1.0]] as [$col, $sx, $sy]) {
    $outputs = $vectorFn(new Dual(2.0, $sx), new Dual(3.0, $sy));
    foreach ($outputs as $row => $output) {
        $jac[$row][$col] = $output->derivative();
    }
}
echo "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n";
for ($row = 0; $row < 3; $row++) {
    echo '  [' . cs($jac[$row][0]) . ', ' . cs($jac[$row][1]) . "]\n";
}

echo "\n=================== Hessian via second-order forward mode (HyperDual) ===================\n";
// One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
[$mx, $my] = [1.0, 1.0];  // the Rosenbrock function's minimum
$seeds = [[1.0, 0.0], [0.0, 1.0]];
$h = [[0.0, 0.0], [0.0, 0.0]];
for ($i = 0; $i < 2; $i++) {
    for ($j = 0; $j < 2; $j++) {
        [$e1, $e2] = [$seeds[$i], $seeds[$j]];
        $h[$i][$j] = rosenbrockHyperDual(new HyperDual($mx, $e1[0], $e2[0], 0.0), new HyperDual($my, $e1[1], $e2[1], 0.0))
            ->eps1eps2();
    }
}
echo "Hessian of the Rosenbrock function at its minimum (1,1):\n";
for ($i = 0; $i < 2; $i++) {
    echo '  [' . cs($h[$i][0]) . ', ' . cs($h[$i][1]) . "]\n";
}

echo "\n=================== Gradient descent driven by reverse-mode gradients ===================\n";
$point = [-1.2, 1.0];  // the classic Rosenbrock start
const LEARNING_RATE = 0.001;
const N_STEPS = 2000;
for ($step = 0; $step < N_STEPS; $step++) {
    $tape = new Tape();
    $vx = new Var_($tape, $point[0]);
    $vy = new Var_($tape, $point[1]);
    $vf = rosenbrockVar($tape, $vx, $vy);
    $grad = $tape->backward($vf);
    $loss = $vf->value();
    $point[0] -= LEARNING_RATE * adjoint($grad, $vx);
    $point[1] -= LEARNING_RATE * adjoint($grad, $vy);
    if ($step === 0 || $step === N_STEPS - 1) {
        echo "step $step: loss = " . cs($loss) . ', x = [' . cs($point[0]) . ', ' . cs($point[1]) . "]\n";
    }
}
echo "(true minimum is at [1, 1] with loss 0)\n";
