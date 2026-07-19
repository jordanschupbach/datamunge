use strict;
use warnings;

use Datamunge;

# Note: the C++ autodiff_ex.cpp additionally demonstrates datamunge::autodiff::derivative/
# gradient_forward/gradient_reverse/jacobian_forward/hessian -- generic C++ template driver
# functions that work over any callable, but templates can't cross the SWIG boundary, so
# they aren't bound in any scripting language. This port demonstrates the same ideas
# (forward-mode gradients via repeated single-seed Dual evaluations, reverse-mode gradients
# via one Tape/Var backward pass, Hessian entries via HyperDual) using only the bound
# named-method API.

sub rosenbrock_dual {
  my ($x0, $x1) = @_;
  my $a = Datamunge::Dual->new(1.0, 0.0)->subtract($x0);
  my $b = $x1->subtract($x0->multiply($x0));
  return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

sub rosenbrock_value {
  my ($x0, $x1) = @_;
  my $a = 1.0 - $x0;
  my $b = $x1 - $x0 * $x0;
  return $a * $a + 100.0 * $b * $b;
}

print "=================== Forward mode: scalar derivative ===================\n";
my $x0 = Datamunge::Dual->new(2.0, 1.0);  # seed derivative = 1 to read df/dx directly
my $f = $x0->multiply($x0)->multiply($x0)->subtract($x0->multiply_scalar(2.0));  # x^3 - 2x
print "f(x) = x^3 - 2x, f'(2) = " . $f->derivative() . " (exact: 10)\n";

print "\n=================== Reverse mode: build a graph by hand ===================\n";
my $tape = Datamunge::Tape->new();
my $a = Datamunge::Var->new($tape, 2.0);
my $b = Datamunge::Var->new($tape, 3.0);
my $y = $a->multiply($b)->add($a->sin());
print "y = a*b + sin(a) at a=2, b=3 -> y = " . $y->value() . "\n";
my $adjoint = $tape->backward($y);
print "dy/da = " . $adjoint->[$a->index()] . " (exact: b + cos(a))\n";
print "dy/db = " . $adjoint->[$b->index()] . " (exact: a)\n";

print "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n";
my $px = 0.0;
my $py = 0.0;

# Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
my $gx = rosenbrock_dual(Datamunge::Dual->new($px, 1.0), Datamunge::Dual->new($py, 0.0))->derivative();
my $gy = rosenbrock_dual(Datamunge::Dual->new($px, 0.0), Datamunge::Dual->new($py, 1.0))->derivative();

# Reverse mode: one Tape/Var pass computes every partial at once.
my $tape2 = Datamunge::Tape->new();
my $vx = Datamunge::Var->new($tape2, $px);
my $vy = Datamunge::Var->new($tape2, $py);
my $va = Datamunge::Var->new($tape2, 1.0)->subtract($vx);
my $vb = $vy->subtract($vx->multiply($vx));
my $vf = $va->multiply($va)->add($vb->multiply($vb)->multiply_scalar(100.0));
my $grad_rev = $tape2->backward($vf);

print "f(0,0) = " . rosenbrock_value($px, $py) . "\n";
print "gradient (forward mode): [$gx, $gy]\n";
print "gradient (reverse mode): [" . $grad_rev->[$vx->index()] . ", " . $grad_rev->[$vy->index()] . "]\n";

print "\n=================== Jacobian of a vector-valued function ===================\n";
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
my $vx_val = 2.0;
my $vy_val = 3.0;

sub vector_fn {
  my ($x, $y) = @_;
  return [$x->multiply($x), $x->multiply($y), $y->multiply($y)->multiply($y)];
}

my @jac = ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]);
my @cols = ([0, 1.0, 0.0], [1, 0.0, 1.0]);
for my $c (@cols) {
  my ($col, $seed_x, $seed_y) = @$c;
  my $outputs = vector_fn(Datamunge::Dual->new($vx_val, $seed_x), Datamunge::Dual->new($vy_val, $seed_y));
  for (my $row = 0; $row < 3; $row++) {
    $jac[$row][$col] = $outputs->[$row]->derivative();
  }
}
print "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n";
for my $row (@jac) { print "  [" . join(", ", @$row) . "]\n"; }

print "\n=================== Hessian via second-order forward mode (HyperDual) ===================\n";
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
my $mx = 1.0;
my $my = 1.0;  # the Rosenbrock function's minimum

sub rosenbrock_hyperdual {
  my ($x0, $x1) = @_;
  my $a = Datamunge::HyperDual->new(1.0, 0.0, 0.0, 0.0)->subtract($x0);
  my $b = $x1->subtract($x0->multiply($x0));
  return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

my @seeds = ([1.0, 0.0], [0.0, 1.0]);
my @h = ([0.0, 0.0], [0.0, 0.0]);
for (my $i = 0; $i < 2; $i++) {
  for (my $j = 0; $j < 2; $j++) {
    my ($e1x, $e1y) = @{$seeds[$i]};
    my ($e2x, $e2y) = @{$seeds[$j]};
    my $hx = Datamunge::HyperDual->new($mx, $e1x, $e2x, 0.0);
    my $hy = Datamunge::HyperDual->new($my, $e1y, $e2y, 0.0);
    $h[$i][$j] = rosenbrock_hyperdual($hx, $hy)->eps1eps2();
  }
}
print "Hessian of the Rosenbrock function at its minimum (1,1):\n";
for my $row (@h) { print "  [" . join(", ", @$row) . "]\n"; }

print "\n=================== Gradient descent driven by reverse-mode gradients ===================\n";
my @point = (-1.2, 1.0);  # the classic Rosenbrock starting point
my $learning_rate = 0.001;
my $n_steps = 2000;
for (my $step = 0; $step < $n_steps; $step++) {
  my $t = Datamunge::Tape->new();
  my $vx2 = Datamunge::Var->new($t, $point[0]);
  my $vy2 = Datamunge::Var->new($t, $point[1]);
  my $va2 = Datamunge::Var->new($t, 1.0)->subtract($vx2);
  my $vb2 = $vy2->subtract($vx2->multiply($vx2));
  my $vf2 = $va2->multiply($va2)->add($vb2->multiply($vb2)->multiply_scalar(100.0));
  my $grad = $t->backward($vf2);
  my $loss = $vf2->value();
  $point[0] -= $learning_rate * $grad->[$vx2->index()];
  $point[1] -= $learning_rate * $grad->[$vy2->index()];
  if ($step == 0 || $step == $n_steps - 1) {
    print "step $step: loss = $loss, x = [" . $point[0] . ", " . $point[1] . "]\n";
  }
}
print "(true minimum is at [1, 1] with loss 0)\n";
