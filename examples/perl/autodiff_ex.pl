# CMD: just TARGET=autodiff_ex run-perl
# Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

# Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
# (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
# bound. This port demonstrates the same ideas with the bound named-method API.
# $var->index is the Var's position in the adjoint vector $tape->backward returns.

sub rosenbrock_dual {
    my ($x0, $x1) = @_;
    my $a = Datamunge::Dual->new(1.0, 0.0)->subtract($x0);
    my $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

sub rosenbrock_value { my ($x0, $x1) = @_; return (1.0 - $x0)**2 + 100.0 * ($x1 - $x0 * $x0)**2 }

# Build the Rosenbrock graph on tape; returns the output Var.
sub rosenbrock_var {
    my ($tape, $x0, $x1) = @_;
    my $a = Datamunge::Var->new($tape, 1.0)->subtract($x0);
    my $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

sub rosenbrock_hyper_dual {
    my ($x0, $x1) = @_;
    my $a = Datamunge::HyperDual->new(1.0, 0.0, 0.0, 0.0)->subtract($x0);
    my $b = $x1->subtract($x0->multiply($x0));
    return $a->multiply($a)->add($b->multiply($b)->multiply_scalar(100.0));
}

sub adjoint { my ($adjoints, $v) = @_; return $adjoints->[$v->index] }

print "=================== Forward mode: scalar derivative ===================\n";
{
    my $x0 = Datamunge::Dual->new(2.0, 1.0);  # seed derivative = 1 to read df/dx directly
    my $f = $x0->multiply($x0)->multiply($x0)->subtract($x0->multiply_scalar(2.0));  # x^3 - 2x
    print "f(x) = x^3 - 2x, f'(2) = ", fmt($f->derivative), " (exact: 10)\n";
}

print "\n=================== Reverse mode: build a graph by hand ===================\n";
{
    my $tape = Datamunge::Tape->new();
    my $a = Datamunge::Var->new($tape, 2.0);
    my $b = Datamunge::Var->new($tape, 3.0);
    my $y = $a->multiply($b)->add($a->sin());
    print "y = a*b + sin(a) at a=2, b=3 -> y = ", fmt($y->value), "\n";
    my $adjoint = $tape->backward($y);
    print "dy/da = ", fmt(adjoint($adjoint, $a)), " (exact: b + cos(a))\n";
    print "dy/db = ", fmt(adjoint($adjoint, $b)), " (exact: a)\n";
}

print "\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================\n";
{
    my ($px, $py) = (0.0, 0.0);
    # Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    my $gx = rosenbrock_dual(Datamunge::Dual->new($px, 1.0), Datamunge::Dual->new($py, 0.0))->derivative;
    my $gy = rosenbrock_dual(Datamunge::Dual->new($px, 0.0), Datamunge::Dual->new($py, 1.0))->derivative;
    # Reverse mode: one Tape/Var pass computes every partial at once.
    my $tape = Datamunge::Tape->new();
    my $vx = Datamunge::Var->new($tape, $px);
    my $vy = Datamunge::Var->new($tape, $py);
    my $grad = $tape->backward(rosenbrock_var($tape, $vx, $vy));
    print "f(0,0) = ", fmt(rosenbrock_value($px, $py)), "\n";
    print "gradient (forward mode): [", fmt($gx), ", ", fmt($gy), "]\n";
    print "gradient (reverse mode): [", fmt(adjoint($grad, $vx)), ", ", fmt(adjoint($grad, $vy)), "]\n";
}

print "\n=================== Jacobian of a vector-valued function ===================\n";
# f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
{
    my $vector_fn = sub { my ($x, $y) = @_; return ($x->multiply($x), $x->multiply($y), $y->multiply($y)->multiply($y)) };
    my @jac;
    for my $seed ([0, 1.0, 0.0], [1, 0.0, 1.0]) {
        my ($col, $sx, $sy) = @$seed;
        my @outputs = $vector_fn->(Datamunge::Dual->new(2.0, $sx), Datamunge::Dual->new(3.0, $sy));
        $jac[$_][$col] = $outputs[$_]->derivative for 0 .. $#outputs;
    }
    print "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:\n";
    print "  [", fmt($jac[$_][0]), ", ", fmt($jac[$_][1]), "]\n" for 0 .. 2;
}

print "\n=================== Hessian via second-order forward mode (HyperDual) ===================\n";
# One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
{
    my ($mx, $my) = (1.0, 1.0);  # the Rosenbrock function's minimum
    my @seeds = ([1.0, 0.0], [0.0, 1.0]);
    my @h;
    for my $i (0 .. 1) {
        for my $j (0 .. 1) {
            my ($e1, $e2) = ($seeds[$i], $seeds[$j]);
            $h[$i][$j] = rosenbrock_hyper_dual(Datamunge::HyperDual->new($mx, $e1->[0], $e2->[0], 0.0),
                                               Datamunge::HyperDual->new($my, $e1->[1], $e2->[1], 0.0))->eps1eps2;
        }
    }
    print "Hessian of the Rosenbrock function at its minimum (1,1):\n";
    print "  [", fmt($h[$_][0]), ", ", fmt($h[$_][1]), "]\n" for 0 .. 1;
}

print "\n=================== Gradient descent driven by reverse-mode gradients ===================\n";
{
    my @point = (-1.2, 1.0);  # the classic Rosenbrock start
    my $learning_rate = 0.001;
    my $n_steps = 2000;
    for my $step (0 .. $n_steps - 1) {
        my $tape = Datamunge::Tape->new();
        my $vx = Datamunge::Var->new($tape, $point[0]);
        my $vy = Datamunge::Var->new($tape, $point[1]);
        my $vf = rosenbrock_var($tape, $vx, $vy);
        my $grad = $tape->backward($vf);
        my $loss = $vf->value;
        $point[0] -= $learning_rate * adjoint($grad, $vx);
        $point[1] -= $learning_rate * adjoint($grad, $vy);
        print "step $step: loss = ", fmt($loss), ", x = [", fmt($point[0]), ", ", fmt($point[1]), "]\n"
            if $step == 0 || $step == $n_steps - 1;
    }
    print "(true minimum is at [1, 1] with loss 0)\n";
}

# Formats a number like C#'s double.ToString(): shortest round-trip digits, E notation outside [1e-4, 1e15).
sub fmt {
    my ($x) = @_;
    return 'NaN' if $x != $x;
    return $x > 0 ? 'Infinity' : '-Infinity' if $x == 9**9**9 || $x == -9**9**9;
    return '0' if $x == 0;
    my $s;
    for my $p (0 .. 16) { $s = sprintf('%.*e', $p, $x); last if $s == $x }
    my ($m, $e) = $s =~ /^(-?[\d.]+)e([-+]\d+)$/;
    $m =~ s/\.?0+$// if $m =~ /\./;
    $e += 0;
    return sprintf('%sE%s%02d', $m, $e < 0 ? '-' : '+', abs $e) if $e < -4 || $e >= 15;
    my $neg = $m =~ s/^-//;
    my $digits = $m =~ s/\.//r;
    my $out;
    if ($e < 0) { $out = '0.' . ('0' x (-$e - 1)) . $digits }
    elsif (length($digits) > $e + 1) { $out = substr($digits, 0, $e + 1) . '.' . substr($digits, $e + 1) }
    else { $out = $digits . ('0' x ($e + 1 - length $digits)) }
    return ($neg ? '-' : '') . $out;
}
