use strict;
use warnings;

use Datamunge;

# ---- Function types, defined via SWIG directors (Perl subclasses of the bound C++
# ---- interfaces). Director callback vector<double> PARAMETERS arrive as real DVector
# objects (index with ->get($i), not ->[$i]); vector<double> RETURN values must be wrapped
# in a real Datamunge::DVector->new([...]), a plain Perl arrayref is rejected. ----

sub dv_to_array {
  my ($dv) = @_;
  my @out;
  for (my $i = 0; $i < $dv->size(); $i++) { push @out, $dv->get($i); }
  return \@out;
}

package QuadraticBowl;
# A plain 3-D quadratic bowl with a known minimum.
our @ISA = ("Datamunge::DifferentiableFunction");

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
    $sum += ($coordinates->get($i) - $self->{target}[$i])**2;
  }
  return $sum;
}

sub gradient {
  my ($self, $coordinates) = @_;
  my @g = map { 2.0 * ($coordinates->get($_) - $self->{target}[$_]) } (0 .. scalar(@{$self->{target}}) - 1);
  return Datamunge::DVector->new(\@g);
}

package RosenbrockFn;
# The classic Rosenbrock "banana" function -- a much harder landscape.
our @ISA = ("Datamunge::DifferentiableFunction");

sub evaluate {
  my ($self, $x) = @_;
  my $a = 1.0 - $x->get(0);
  my $b = $x->get(1) - $x->get(0) * $x->get(0);
  return $a * $a + 100.0 * $b * $b;
}

sub gradient {
  my ($self, $x) = @_;
  my $x0 = $x->get(0);
  my $x1 = $x->get(1);
  return Datamunge::DVector->new([-2.0 * (1.0 - $x0) - 400.0 * $x0 * ($x1 - $x0 * $x0), 200.0 * ($x1 - $x0 * $x0)]);
}

package LinearRegressionLoss;
# Ordinary least squares as a sum of per-example losses -- the textbook case for SGD.
our @ISA = ("Datamunge::DifferentiableSeparableFunction");

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
  for (my $j = 0; $j < $w->size(); $j++) { $sum += $w->get($j) * $self->{x}[$i][$j]; }
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
  my @g = map { 2.0 * $err * $self->{x}[$i][$_] } (0 .. $w->size() - 1);
  return Datamunge::DVector->new(\@g);
}

package BumpyFunction;
# A bumpy, multimodal landscape -- no gradient available, so only a derivative-free
# method (SimulatedAnnealing) can be used here.
our @ISA = ("Datamunge::ArbitraryFunction");

sub evaluate {
  my ($self, $x) = @_;
  my $bowl = ($x->get(0) - 3.0)**2 + ($x->get(1) + 1.0)**2;
  my $ripples = 5.0 * sin($x->get(0)) * cos($x->get(1));
  return $bowl + $ripples;
}

package RastriginFunction;
# The classic Rastrigin function -- highly multimodal (many local minima arranged in a
# regular grid), global minimum f=0 at the origin. A standard torture test for
# population-based methods, since local/gradient-based methods get stuck in the first
# basin they land in.
our @ISA = ("Datamunge::ArbitraryFunction");

sub evaluate {
  my ($self, $x) = @_;
  my $total = 10.0 * $x->size();
  for (my $i = 0; $i < $x->size(); $i++) {
    my $xi = $x->get($i);
    $total += $xi * $xi - 10.0 * cos(2.0 * 3.14159265358979 * $xi);
  }
  return $total;
}

package main;

print "=================== DifferentiableFunction: three optimizers, one bowl ===================\n";
my @target = (4.0, -2.0, 1.0);

my $f = QuadraticBowl->new(\@target);
my $x = Datamunge::DVector->new([0.0, 0.0, 0.0]);
my $gd_options = Datamunge::GradientDescentOptions->new();
$gd_options->swig_step_size_set(0.1);
$gd_options->swig_momentum_set(0.0);
$gd_options->swig_max_iterations_set(1000);
$gd_options->swig_tolerance_set(1e-10);
my $value = Datamunge::GradientDescent->new($gd_options)->optimize($f, $x);
print "GradientDescent: f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = QuadraticBowl->new(\@target);
$x = Datamunge::DVector->new([0.0, 0.0, 0.0]);
$value = Datamunge::Adam->new()->optimize($f, $x);
print "Adam:             f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = QuadraticBowl->new(\@target);
$x = Datamunge::DVector->new([0.0, 0.0, 0.0]);
$value = Datamunge::LBFGS->new()->optimize($f, $x);
print "LBFGS:            f=$value x=[" . join(",", @{dv_to_array($x)}) . "] (converges in far fewer iterations)\n";

print "\n=================== LBFGS on the Rosenbrock function ===================\n";
$f = RosenbrockFn->new();
$x = Datamunge::DVector->new([-1.2, 1.0]);
$value = Datamunge::LBFGS->new()->optimize($f, $x);
print "f=$value x=[" . join(",", @{dv_to_array($x)}) . "] (true minimum: f=0 at [1, 1])\n";

print "\n=================== DifferentiableSeparableFunction: SGD vs. closed-form LM ===================\n";
my $iris = Datamunge::DataFrame::iris();
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

my $w_std = Datamunge::DVector->new([0.0, 0.0, 0.0]);
my $sgd_options = Datamunge::SGDOptions->new();
$sgd_options->swig_step_size_set(0.01);
$sgd_options->swig_max_epochs_set(300);
$sgd_options->swig_batch_size_set(8);
Datamunge::SGD->new($sgd_options)->optimize($loss, $w_std);

# Convert the standardized-space weights back to the original feature scale.
my @w = ($w_std->get(0) - $w_std->get(1) * $mean1 / $std1 - $w_std->get(2) * $mean2 / $std2,
         $w_std->get(1) / $std1, $w_std->get(2) / $std2);
print "SGD weights (intercept, Sepal.Length, Sepal.Width): [" . join(",", @w) . "]\n";

my $lm = Datamunge::LM->new($iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
print "LM  weights (intercept, Sepal.Length, Sepal.Width): [" . join(",", @{$lm->coefficients()}) . "] (closed-form OLS, for comparison)\n";

print "\n=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================\n";
$f = BumpyFunction->new();
$x = Datamunge::DVector->new([0.0, 0.0]);
my $sa_options = Datamunge::SimulatedAnnealingOptions->new();
$sa_options->swig_initial_temperature_set(10.0);
$sa_options->swig_cooling_rate_set(0.999);
$sa_options->swig_max_iterations_set(20000);
$sa_options->swig_step_std_dev_set(0.5);
$value = Datamunge::SimulatedAnnealing->new($sa_options)->optimize($f, $x);
print "f=$value x=[" . join(",", @{dv_to_array($x)}) . "] (found without ever computing a gradient)\n";

print "\n=================== Population-based methods on the Rastrigin function ===================\n";
my @lower = (-5.12, -5.12);
my @upper = (5.12, 5.12);

$f = RastriginFunction->new();
$x = Datamunge::DVector->new([3.0, -4.0]);
my $pso_options = Datamunge::PSOOptions->new();
$pso_options->swig_topology_set("global");
$pso_options->swig_inertia_strategy_set("constant");
$value = Datamunge::PSO->new($pso_options)->optimize($f, $x, \@lower, \@upper);
print "PSO (global topology, constant inertia):    f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = RastriginFunction->new();
$x = Datamunge::DVector->new([3.0, -4.0]);
$pso_options = Datamunge::PSOOptions->new();
$pso_options->swig_topology_set("ring");
$pso_options->swig_inertia_strategy_set("linear_decay");
$value = Datamunge::PSO->new($pso_options)->optimize($f, $x, \@lower, \@upper);
print "PSO (ring topology, linear-decay inertia):  f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = RastriginFunction->new();
$x = Datamunge::DVector->new([3.0, -4.0]);
my $de_options = Datamunge::DEOptions->new();
$de_options->swig_mutation_strategy_set("rand1");
$de_options->swig_crossover_strategy_set("binomial");
$value = Datamunge::DifferentialEvolution->new($de_options)->optimize($f, $x, \@lower, \@upper);
print "DE (rand1/binomial):                        f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = RastriginFunction->new();
$x = Datamunge::DVector->new([3.0, -4.0]);
$de_options = Datamunge::DEOptions->new();
$de_options->swig_mutation_strategy_set("best1");
$de_options->swig_crossover_strategy_set("exponential");
$value = Datamunge::DifferentialEvolution->new($de_options)->optimize($f, $x, \@lower, \@upper);
print "DE (best1/exponential):                     f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = RastriginFunction->new();
$x = Datamunge::DVector->new([3.0, -4.0]);
my $ga_options = Datamunge::GAOptions->new();
$ga_options->swig_selection_strategy_set("tournament");
$ga_options->swig_crossover_strategy_set("blend");
$value = Datamunge::GeneticAlgorithm->new($ga_options)->optimize($f, $x, \@lower, \@upper);
print "GA (tournament/blend, elitism on):           f=$value x=[" . join(",", @{dv_to_array($x)}) . "]\n";

$f = RastriginFunction->new();
$x = Datamunge::DVector->new([3.0, -4.0]);
$ga_options = Datamunge::GAOptions->new();
$ga_options->swig_selection_strategy_set("rank");
$ga_options->swig_crossover_strategy_set("uniform");
$ga_options->swig_elitism_set(0);
$value = Datamunge::GeneticAlgorithm->new($ga_options)->optimize($f, $x, \@lower, \@upper);
print "GA (rank/uniform, elitism off):              f=$value x=[" . join(",", @{dv_to_array($x)}) . "] (true minimum: f=0 at [0, 0])\n";
