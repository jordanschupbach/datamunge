# CMD: just TARGET=optim_advanced_ex run-perl
# Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
# AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Perl.
#
# Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
# EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
# ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
# implement the objective. From Perl, subclass the interface class
# (use parent -norequire, 'Datamunge::HessianFunction') and define its virtual methods; the
# optimizer then calls the Perl methods. Coordinates are updated in place through an array
# reference (the C++ parameter is a non-const std::vector<double>&).
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

use List::Util qw(min);

{
    # Newton's objective f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
    package Quadratic;
    use parent -norequire, 'Datamunge::HessianFunction';

    sub evaluate { my ($self, $c) = @_; return ($c->[0] - 1)**2 + 10 * ($c->[1] + 2)**2 }
    sub gradient { my ($self, $c) = @_; return [2 * ($c->[0] - 1), 20 * ($c->[1] + 2)] }
    sub hessian { return [[2.0, 0.0], [0.0, 20.0]] }
}
{
    # Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
    package Exponential;
    use parent -norequire, 'Datamunge::ResidualFunction';

    sub new {
        my ($class, $ts) = @_;
        my $self = $class->SUPER::new();
        $self->{ts} = $ts;
        return $self;
    }

    sub residuals {
        my ($self, $p) = @_;
        return [map { $p->[0] * exp($p->[1] * $_) - 2 * exp(0.5 * $_) } @{ $self->{ts} }];
    }

    sub jacobian {
        my ($self, $p) = @_;
        return [map { [exp($p->[1] * $_), $p->[0] * $_ * exp($p->[1] * $_)] } @{ $self->{ts} }];
    }
}
{
    # Leaves gradient() and hessian() out.
    package Incomplete;
    use parent -norequire, 'Datamunge::HessianFunction';

    sub evaluate { return 0.0 }
}

# An options object with the given fields set (the C++ structs have no keyword constructor).
sub options {
    my ($class, %fields) = @_;
    my $options = $class->new();
    $options->$_($fields{$_}) for sort keys %fields;
    return $options;
}

# C#'s object.ToString() for the bound classes: the class name.
sub describe { my ($object) = @_; return (ref($object) =~ s/^.*:://r) . "()" }

# C#'s "G<digits>" format.
sub fmt_g { my ($x, $digits) = @_; return sprintf('%.*g', $digits, $x) =~ s/e/E/r }

print "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================\n";
# Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
# then check the acquisition function favors an unexplored region.
{
    my @xs = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0);
    my @ys = map { ($_ - 1) * ($_ - 1) } @xs;
    my $surrogate = Datamunge::RbfGaussianProcessSurrogate->new(1.0, 1e-6);
    $surrogate->fit([map { [$_] } @xs], \@ys);
    my $incumbent = min(0.0, @ys);  # best observed value so far
    print "incumbent (best observed y): ", fmt($incumbent), "\n";
    printf "acquisition(x=%.2f) = %.6f\n", $_, $surrogate->acquisition([$_], $incumbent) for 0.25, 1.0, 2.5;
}

print "\n=================== Options structs + optimizer construction (all 10 new classes) ===================\n";
# Every new Options struct and optimizer class constructs cleanly, proving
# the binding for the whole batch is wired correctly.
my $pg = Datamunge::ProximalGradient->new(options('Datamunge::ProximalGradientOptions', step_size => 0.05));
print "ProximalGradient: ", describe($pg), "\n";
my $fista = Datamunge::Fista->new(Datamunge::FistaOptions->new());
print "FISTA: ", describe($fista), "\n";
my $lm = Datamunge::LevenbergMarquardt->new(options('Datamunge::LevenbergMarquardtOptions', max_iterations => 200));
print "LevenbergMarquardt: ", describe($lm), "\n";
my $newton = Datamunge::Newton->new(Datamunge::NewtonOptions->new());
print "Newton: ", describe($newton), "\n";
my $trn = Datamunge::TrustRegionNewton->new(Datamunge::TrustRegionNewtonOptions->new());
print "TrustRegionNewton: ", describe($trn), "\n";
my $al = Datamunge::AugmentedLagrangian->new(Datamunge::AugmentedLagrangianOptions->new());
print "AugmentedLagrangian: ", describe($al), "\n";
my $sqp = Datamunge::Sqp->new(Datamunge::SqpOptions->new());
print "SQP: ", describe($sqp), "\n";
my $ip = Datamunge::InteriorPoint->new(Datamunge::InteriorPointOptions->new());
print "InteriorPoint: ", describe($ip), "\n";
my $bo = Datamunge::BayesianOptimization->new(options('Datamunge::BayesianOptimizationOptions', initial_samples => 8));
print "BayesianOptimization: ", describe($bo), "\n";

print "\n=================== Perl objectives (subclassing the C++ interfaces) ===================\n";
# Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
{
    my @start = (5.0, 5.0);
    my $value = $newton->optimize(Quadratic->new(), \@start);
    printf "Newton: f = %s at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n", fmt_g($value, 6), @start;
}

# Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
{
    my @parameters = (1.0, 0.1);
    my $sse = $lm->optimize(Exponential->new([0.0, 0.5, 1.0, 1.5, 2.0]), \@parameters);
    printf "LevenbergMarquardt: residual = %s, a = %.6f, b = %.6f (exact: 2, 0.5)\n", fmt_g($sse, 3), @parameters;
}

# An interface method a subclass leaves out dies when the optimizer calls it.
{
    my @origin = (0.0, 0.0);
    my $ok = eval { $newton->optimize(Incomplete->new(), \@origin); 1 };
    my $err = $@;
    printf "missing gradient(): ok=%s, error mentions it: %s\n", $ok ? "True" : "False", (!$ok && "$err" =~ /gradient/) ? "True" : "False";
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
