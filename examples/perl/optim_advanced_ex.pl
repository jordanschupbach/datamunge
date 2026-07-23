use strict;
use warnings;

use Datamunge;

# ---- New optim function-type interfaces (ProximalFunction, HessianFunction,
# ---- EqualityConstrainedFunction, InequalityConstrainedFunction, ResidualFunction,
# ---- BayesianSurrogate), exercised the same way as the existing directors in optim_ex.pl:
# director callback vector<double> PARAMETERS arrive as real DVector objects (index with
# ->get($i), not ->[$i]); vector<double> RETURN values must be wrapped in a real
# Datamunge::DVector->new([...]); vector<vector<double>> RETURN values must be wrapped in a
# real Datamunge::DVectorVector, built by push()-ing either DVector objects or plain
# arrayrefs (push() accepts a plain Perl arrayref directly for the inner vector<double> -- no
# need to wrap each row in DVector->new first). ----

sub dv_to_array {
  my ($dv) = @_;
  my @out;
  for (my $i = 0; $i < $dv->size(); $i++) { push @out, $dv->get($i); }
  return \@out;
}

package QuadraticBowlHessian;
# A simple 2-D quadratic bowl with a known minimum -- exercises HessianFunction, the new
# interface Newton/TrustRegionNewton consume.
our @ISA = ("Datamunge::HessianFunction");

sub new {
  my ($class, $target) = @_;
  my $self = $class->SUPER::new();
  $self->{target} = $target;
  return $self;
}

sub evaluate {
  my ($self, $coordinates) = @_;
  my $dx = $coordinates->get(0) - $self->{target}[0];
  my $dy = $coordinates->get(1) - $self->{target}[1];
  return $dx * $dx + $dy * $dy;
}

sub gradient {
  my ($self, $coordinates) = @_;
  my $dx = $coordinates->get(0) - $self->{target}[0];
  my $dy = $coordinates->get(1) - $self->{target}[1];
  return Datamunge::DVector->new([2.0 * $dx, 2.0 * $dy]);
}

sub hessian {
  my ($self, $coordinates) = @_;
  my $h = Datamunge::DVectorVector->new();
  $h->push([2.0, 0.0]);
  $h->push([0.0, 2.0]);
  return $h;
}

package ExponentialDecayResidual;
# A*exp(-k*t)+c curve-fit residuals -- exercises ResidualFunction, the standalone
# (non-ArbitraryFunction) interface LevenbergMarquardt consumes.
our @ISA = ("Datamunge::ResidualFunction");

sub new {
  my ($class, $t, $y) = @_;
  my $self = $class->SUPER::new();
  $self->{t} = $t;
  $self->{y} = $y;
  return $self;
}

sub residuals {
  my ($self, $coordinates) = @_;
  my $A = $coordinates->get(0);
  my $k = $coordinates->get(1);
  my $c = $coordinates->get(2);
  my @r;
  for (my $i = 0; $i < scalar(@{$self->{t}}); $i++) {
    my $ti = $self->{t}[$i];
    my $model = $A * exp(-$k * $ti) + $c;
    push @r, $model - $self->{y}[$i];
  }
  return Datamunge::DVector->new(\@r);
}

sub jacobian {
  my ($self, $coordinates) = @_;
  my $A = $coordinates->get(0);
  my $k = $coordinates->get(1);
  my $j = Datamunge::DVectorVector->new();
  for (my $i = 0; $i < scalar(@{$self->{t}}); $i++) {
    my $ti = $self->{t}[$i];
    my $e = exp(-$k * $ti);
    $j->push([$e, -$A * $ti * $e, 1.0]);
  }
  return $j;
}

package main;

print "=================== HessianFunction: Newton on a 2-D quadratic bowl ===================\n";
my @target = (3.0, -2.0);
my $f = QuadraticBowlHessian->new(\@target);
my $x = Datamunge::DVector->new([0.0, 0.0]);
my $newton_options = Datamunge::NewtonOptions->new();
$newton_options->swig_max_iterations_set(100);
$newton_options->swig_tolerance_set(1e-10);
my $value = Datamunge::Newton->new($newton_options)->optimize($f, $x);
print "Newton: f=$value x=[" . join(",", @{dv_to_array($x)}) . "] (true minimum: f=0 at [3, -2])\n";

print "\n=================== ResidualFunction: LevenbergMarquardt curve fit ===================\n";
# Synthetic data generated from the known parameters A=5, k=0.7, c=1, noiseless -- LM should
# recover them (near-)exactly from a nearby starting guess.
my $A_true = 5.0;
my $k_true = 0.7;
my $c_true = 1.0;
my @t = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0);
my @y = map { $A_true * exp(-$k_true * $_) + $c_true } @t;

my $residual_fn = ExponentialDecayResidual->new(\@t, \@y);
my $params = Datamunge::DVector->new([1.0, 1.0, 0.0]);
my $lm_options = Datamunge::LevenbergMarquardtOptions->new();
$lm_options->swig_max_iterations_set(200);
$lm_options->swig_tolerance_set(1e-10);
$value = Datamunge::LevenbergMarquardt->new($lm_options)->optimize($residual_fn, $params);
print "LevenbergMarquardt: sum_sq_residuals=$value params(A,k,c)=[" . join(",", @{dv_to_array($params)}) . "]\n";
print "(true parameters: A=$A_true, k=$k_true, c=$c_true)\n";

print "\n=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================\n";
package Bowl1D;
our @ISA = ("Datamunge::ArbitraryFunction");

sub evaluate {
  my ($self, $x) = @_;
  my $d = $x->get(0) - 1.75;
  return $d * $d + 0.1;
}

package main;

my $bo_f = Bowl1D->new();
my $bo_x = Datamunge::DVector->new([0.0]);
my @lower = (-5.0);
my @upper = (5.0);
my $bo_options = Datamunge::BayesianOptimizationOptions->new();
$bo_options->swig_initial_samples_set(8);
$bo_options->swig_max_iterations_set(40);
$bo_options->swig_seed_set(42);
my $surrogate = Datamunge::RBFGaussianProcessSurrogate->new(1.0, 1e-6);
$value = Datamunge::BayesianOptimization->new($bo_options)->optimize($bo_f, $bo_x, \@lower, \@upper, $surrogate);
print "BayesianOptimization: f=$value x=[" . join(",", @{dv_to_array($bo_x)}) . "] (true minimum: f=0.1 at [1.75])\n";
