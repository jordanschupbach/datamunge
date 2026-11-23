# CMD: just TARGET=optim_ex run-perl
# Optimizers driven by Perl objectives: gradient-based, stochastic (SGD), derivative-free and population-based.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

# ---- Function types: Perl subclasses of the bound C++ interfaces (use parent -norequire)
# ---- whose methods the optimizers call. std::vector<double> parameters arrive as array
# ---- references, returned vectors are array references, and the optimizers update the
# ---- coordinates array reference in place (a non-const std::vector<double>&). ----

package QuadraticBowl;
# A plain 3-D quadratic bowl with a known minimum.
use parent -norequire, 'Datamunge::DifferentiableFunction';

sub new {
  my ($class, $target) = @_;
  my $self = $class->SUPER::new();
  $self->{target} = $target;
  return $self;
}

sub evaluate {
  my ($self, $coordinates) = @_;
  my $sum = 0.0;
  for (my $i = 0; $i < scalar(@{$self->{target}}); $i++) {
    $sum += ($coordinates->[$i] - $self->{target}[$i])**2;
  }
  return $sum;
}

sub gradient {
  my ($self, $coordinates) = @_;
  my @g = map { 2.0 * ($coordinates->[$_] - $self->{target}[$_]) } (0 .. scalar(@{$self->{target}}) - 1);
  return \@g;
}

package RosenbrockFn;
# The classic Rosenbrock "banana" function -- a much harder landscape.
use parent -norequire, 'Datamunge::DifferentiableFunction';

sub evaluate {
  my ($self, $x) = @_;
  my $a = 1.0 - $x->[0];
  my $b = $x->[1] - $x->[0] * $x->[0];
  return $a * $a + 100.0 * $b * $b;
}

sub gradient {
  my ($self, $x) = @_;
  my $x0 = $x->[0];
  my $x1 = $x->[1];
  return [-2.0 * (1.0 - $x0) - 400.0 * $x0 * ($x1 - $x0 * $x0), 200.0 * ($x1 - $x0 * $x0)];
}

package LinearRegressionLoss;
# Ordinary least squares as a sum of per-example losses -- the textbook case for SGD.
use parent -norequire, 'Datamunge::DifferentiableSeparableFunction';

sub new {
  my ($class, $x, $y) = @_;
  my $self = $class->SUPER::new();
  $self->{x} = $x;
  $self->{y} = $y;
  return $self;
}

sub num_functions {
  my ($self) = @_;
  return scalar(@{$self->{y}});
}

sub predict {
  my ($self, $w, $i) = @_;
  my $sum = 0.0;
  for (my $j = 0; $j < scalar(@$w); $j++) { $sum += $w->[$j] * $self->{x}[$i][$j]; }
  return $sum;
}

sub evaluate_term {
  my ($self, $w, $i) = @_;
  my $err = $self->predict($w, $i) - $self->{y}[$i];
  return $err * $err;
}

sub gradient_term {
  my ($self, $w, $i) = @_;
  my $err = $self->predict($w, $i) - $self->{y}[$i];
  my @g = map { 2.0 * $err * $self->{x}[$i][$_] } (0 .. scalar(@$w) - 1);
  return \@g;
}

package BumpyFunction;
# A bumpy, multimodal landscape -- no gradient available, so only a derivative-free
# method (SimulatedAnnealing) can be used here.
use parent -norequire, 'Datamunge::ArbitraryFunction';

sub evaluate {
  my ($self, $x) = @_;
  my $bowl = ($x->[0] - 3.0)**2 + ($x->[1] + 1.0)**2;
  my $ripples = 5.0 * sin($x->[0]) * cos($x->[1]);
  return $bowl + $ripples;
}

package RastriginFunction;
# The classic Rastrigin function -- highly multimodal (many local minima arranged in a
# regular grid), global minimum f=0 at the origin. A standard torture test for
# population-based methods, since local/gradient-based methods get stuck in the first
# basin they land in.
use parent -norequire, 'Datamunge::ArbitraryFunction';

sub evaluate {
  my ($self, $x) = @_;
  my $total = 10.0 * scalar(@$x);
  for (my $i = 0; $i < scalar(@$x); $i++) {
    my $xi = $x->[$i];
    $total += $xi * $xi - 10.0 * cos(2.0 * 3.14159265358979 * $xi);
  }
  return $total;
}

package main;

print "=================== DifferentiableFunction: three optimizers, one bowl ===================\n";
my @target = (4.0, -2.0, 1.0);

my $f = QuadraticBowl->new(\@target);
my $x = [0.0, 0.0, 0.0];
my $gd_options = Datamunge::GradientDescentOptions->new();
$gd_options->step_size(0.1);
$gd_options->momentum(0.0);
$gd_options->max_iterations(1000);
$gd_options->tolerance(1e-10);
my $value = Datamunge::GradientDescent->new($gd_options)->optimize($f, $x);
print "GradientDescent: f=$value x=[" . join(",", @$x) . "]\n";

$f = QuadraticBowl->new(\@target);
$x = [0.0, 0.0, 0.0];
$value = Datamunge::Adam->new()->optimize($f, $x);
print "Adam:             f=$value x=[" . join(",", @$x) . "]\n";

$f = QuadraticBowl->new(\@target);
$x = [0.0, 0.0, 0.0];
$value = Datamunge::Lbfgs->new()->optimize($f, $x);
print "LBFGS:            f=$value x=[" . join(",", @$x) . "] (converges in far fewer iterations)\n";

print "\n=================== LBFGS on the Rosenbrock function ===================\n";
$f = RosenbrockFn->new();
$x = [-1.2, 1.0];
$value = Datamunge::Lbfgs->new()->optimize($f, $x);
print "f=$value x=[" . join(",", @$x) . "] (true minimum: f=0 at [1, 1])\n";

print "\n=================== DifferentiableSeparableFunction: SGD vs. closed-form LM ===================\n";
my $iris = Datamunge::Data::iris();
my @sepal_length = map { $iris->numeric_at("Sepal.Length", $_) } (0 .. $iris->nrows() - 1);
my @sepal_width = map { $iris->numeric_at("Sepal.Width", $_) } (0 .. $iris->nrows() - 1);
my @petal_length = map { $iris->numeric_at("Petal.Length", $_) } (0 .. $iris->nrows() - 1);

# SGD with a single constant step size converges far faster (and far more reliably) on
# standardized features -- unnormalized predictors of very different scales give the loss an
# ill-conditioned Hessian, which plain constant-step SGD handles poorly. Standard practice.
sub mean_of { my ($v) = @_; my $s = 0; $s += $_ for @$v; return $s / scalar(@$v); }
sub stddev_of {
  my ($v, $mean) = @_;
  my $s = 0;
  $s += ($_ - $mean)**2 for @$v;
  return sqrt($s / scalar(@$v));
}

my $mean1 = mean_of(\@sepal_length);
my $std1 = stddev_of(\@sepal_length, $mean1);
my $mean2 = mean_of(\@sepal_width);
my $std2 = stddev_of(\@sepal_width, $mean2);

my @x_std = map { [1.0, ($sepal_length[$_] - $mean1) / $std1, ($sepal_width[$_] - $mean2) / $std2] } (0 .. $iris->nrows() - 1);
my $loss = LinearRegressionLoss->new(\@x_std, \@petal_length);

my $w_std = [0.0, 0.0, 0.0];
my $sgd_options = Datamunge::SgdOptions->new();
$sgd_options->step_size(0.01);
$sgd_options->max_epochs(300);
$sgd_options->batch_size(8);
Datamunge::Sgd->new($sgd_options)->optimize($loss, $w_std);

# Convert the standardized-space weights back to the original feature scale.
my @w = ($w_std->[0] - $w_std->[1] * $mean1 / $std1 - $w_std->[2] * $mean2 / $std2,
         $w_std->[1] / $std1, $w_std->[2] / $std2);
print "SGD weights (intercept, Sepal.Length, Sepal.Width): [" . join(",", @w) . "]\n";

my $lm = Datamunge::Lm->new($iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
print "LM  weights (intercept, Sepal.Length, Sepal.Width): [" . join(",", @{$lm->coefficients()}) . "] (closed-form OLS, for comparison)\n";

print "\n=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================\n";
$f = BumpyFunction->new();
$x = [0.0, 0.0];
my $sa_options = Datamunge::SimulatedAnnealingOptions->new();
$sa_options->initial_temperature(10.0);
$sa_options->cooling_rate(0.999);
$sa_options->max_iterations(20000);
$sa_options->step_std_dev(0.5);
$value = Datamunge::SimulatedAnnealing->new($sa_options)->optimize($f, $x);
print "f=$value x=[" . join(",", @$x) . "] (found without ever computing a gradient)\n";

print "\n=================== Population-based methods on the Rastrigin function ===================\n";
my @lower = (-5.12, -5.12);
my @upper = (5.12, 5.12);

$f = RastriginFunction->new();
$x = [3.0, -4.0];
my $pso_options = Datamunge::PsoOptions->new();
$pso_options->topology("global");
$pso_options->inertia_strategy("constant");
$value = Datamunge::Pso->new($pso_options)->optimize($f, $x, \@lower, \@upper);
print "PSO (global topology, constant inertia):    f=$value x=[" . join(",", @$x) . "]\n";

$f = RastriginFunction->new();
$x = [3.0, -4.0];
$pso_options = Datamunge::PsoOptions->new();
$pso_options->topology("ring");
$pso_options->inertia_strategy("linear_decay");
$value = Datamunge::Pso->new($pso_options)->optimize($f, $x, \@lower, \@upper);
print "PSO (ring topology, linear-decay inertia):  f=$value x=[" . join(",", @$x) . "]\n";

$f = RastriginFunction->new();
$x = [3.0, -4.0];
my $de_options = Datamunge::DeOptions->new();
$de_options->mutation_strategy("rand1");
$de_options->crossover_strategy("binomial");
$value = Datamunge::DifferentialEvolution->new($de_options)->optimize($f, $x, \@lower, \@upper);
print "DE (rand1/binomial):                        f=$value x=[" . join(",", @$x) . "]\n";

$f = RastriginFunction->new();
$x = [3.0, -4.0];
$de_options = Datamunge::DeOptions->new();
$de_options->mutation_strategy("best1");
$de_options->crossover_strategy("exponential");
$value = Datamunge::DifferentialEvolution->new($de_options)->optimize($f, $x, \@lower, \@upper);
print "DE (best1/exponential):                     f=$value x=[" . join(",", @$x) . "]\n";

$f = RastriginFunction->new();
$x = [3.0, -4.0];
my $ga_options = Datamunge::GaOptions->new();
$ga_options->selection_strategy("tournament");
$ga_options->crossover_strategy("blend");
$value = Datamunge::GeneticAlgorithm->new($ga_options)->optimize($f, $x, \@lower, \@upper);
print "GA (tournament/blend, elitism on):           f=$value x=[" . join(",", @$x) . "]\n";

$f = RastriginFunction->new();
$x = [3.0, -4.0];
$ga_options = Datamunge::GaOptions->new();
$ga_options->selection_strategy("rank");
$ga_options->crossover_strategy("uniform");
$ga_options->elitism(0);
$value = Datamunge::GeneticAlgorithm->new($ga_options)->optimize($f, $x, \@lower, \@upper);
print "GA (rank/uniform, elitism off):              f=$value x=[" . join(",", @$x) . "] (true minimum: f=0 at [0, 0])\n";
