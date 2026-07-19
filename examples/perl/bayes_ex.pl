use strict;
use warnings;

use Datamunge;

# Note: the C++ bayes_ex.cpp builds its models with datamunge::bayes::AutodiffModel, which
# wraps a Tape/Var lambda for automatic differentiation of the log-posterior -- but
# AutodiffModel is explicitly C++-only (never bound via SWIG, in any language), since a
# scripting-language callback can't be traced by the C++ autodiff tape. This port instead
# subclasses optim::DifferentiableFunction directly (a director, available in this binding)
# and supplies the log-posterior and its gradient by hand -- exactly what AutodiffModel would
# have computed automatically. MAP/HMC/NUTS consume this identically either way.
#
# Director callback vector<double> PARAMETERS arrive as real DVector objects (->get($i));
# vector<double> RETURN values must be wrapped in Datamunge::DVector->new([...]).

sub normal_lpdf {
  my ($x, $mean, $sd) = @_;
  return -0.5 * log(2.0 * 3.14159265358979 * $sd * $sd) - ($x - $mean)**2 / (2.0 * $sd * $sd);
}

sub normal_dlpdf {
  my ($x, $mean, $sd) = @_;
  return -($x - $mean) / ($sd * $sd);
}

sub sigmoid {
  my ($z) = @_;
  return 1.0 / (1.0 + exp(-$z));
}

sub bernoulli_logit_lpmf {
  my ($y, $eta) = @_;
  # log(1+exp(eta)), computed in a numerically stable way.
  return $y * $eta - ($eta > 0 ? $eta + log(1.0 + exp(-$eta)) : log(1.0 + exp($eta)));
}

sub mean_of {
  my ($samples, $dim) = @_;
  my $s = 0.0;
  for (my $i = 0; $i < $samples->size(); $i++) { $s += $samples->get($i)->get($dim); }
  return $s / $samples->size();
}

sub sd_of {
  my ($samples, $dim, $mean) = @_;
  my $s = 0.0;
  for (my $i = 0; $i < $samples->size(); $i++) { $s += ($samples->get($i)->get($dim) - $mean)**2; }
  return sqrt($s / ($samples->size() - 1));
}

package ConjugateModel;
our @ISA = ("Datamunge::DifferentiableFunction");

sub new {
  my ($class, $y, $mu0, $tau0, $sigma) = @_;
  my $self = $class->SUPER::new();
  $self->{y} = $y;
  $self->{mu0} = $mu0;
  $self->{tau0} = $tau0;
  $self->{sigma} = $sigma;
  return $self;
}

sub evaluate {
  my ($self, $params) = @_;
  my $mu = $params->get(0);
  my $lp = main::normal_lpdf($mu, $self->{mu0}, $self->{tau0});
  for my $yi (@{$self->{y}}) { $lp += main::normal_lpdf($yi, $mu, $self->{sigma}); }
  return $lp;
}

sub gradient {
  my ($self, $params) = @_;
  my $mu = $params->get(0);
  my $d = main::normal_dlpdf($mu, $self->{mu0}, $self->{tau0});
  for my $yi (@{$self->{y}}) { $d += -main::normal_dlpdf($yi, $mu, $self->{sigma}); }
  return Datamunge::DVector->new([$d]);
}

package LogisticModel;
our @ISA = ("Datamunge::DifferentiableFunction");

sub new {
  my ($class, $x, $y) = @_;
  my $self = $class->SUPER::new();
  $self->{x} = $x;
  $self->{y} = $y;
  return $self;
}

sub evaluate {
  my ($self, $params) = @_;
  my $b0 = $params->get(0);
  my $b1 = $params->get(1);
  my $lp = main::normal_lpdf($b0, 0.0, 10.0) + main::normal_lpdf($b1, 0.0, 10.0);
  for (my $i = 0; $i < scalar(@{$self->{x}}); $i++) {
    $lp += main::bernoulli_logit_lpmf($self->{y}[$i], $b0 + $b1 * $self->{x}[$i]);
  }
  return $lp;
}

sub gradient {
  my ($self, $params) = @_;
  my $b0 = $params->get(0);
  my $b1 = $params->get(1);
  my $d0 = main::normal_dlpdf($b0, 0.0, 10.0);
  my $d1 = main::normal_dlpdf($b1, 0.0, 10.0);
  for (my $i = 0; $i < scalar(@{$self->{x}}); $i++) {
    my $resid = $self->{y}[$i] - main::sigmoid($b0 + $b1 * $self->{x}[$i]);
    $d0 += $resid;
    $d1 += $resid * $self->{x}[$i];
  }
  return Datamunge::DVector->new([$d0, $d1]);
}

package main;

print "=================== Normal-Normal conjugate model: MAP, HMC, NUTS vs. the exact posterior ===================\n";
my @y = (2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 1.95);
my $sigma = 1.0;
my $mu0 = 0.0;
my $tau0 = 5.0;

my $n = scalar(@y);
my $precision_post = 1.0 / ($tau0 * $tau0) + $n / ($sigma * $sigma);
my $y_sum = 0.0;
$y_sum += $_ for @y;
my $exact_mean = ($mu0 / ($tau0 * $tau0) + $y_sum / ($sigma * $sigma)) / $precision_post;
my $exact_sd = sqrt(1.0 / $precision_post);
print "Exact posterior: N($exact_mean, $exact_sd^2)\n\n";

my $conjugate_model = ConjugateModel->new(\@y, $mu0, $tau0, $sigma);

# MAP().optimize() mutates its coordinates argument in place, so it needs a real DVector
# (a plain Perl arrayref works for by-value/const-ref arguments, but not this in/out one).
my $coords = Datamunge::DVector->new([0.0]);
my $log_post = Datamunge::MAP->new()->optimize($conjugate_model, $coords);
print "MAP:  mu = " . $coords->get(0) . " (log-posterior = $log_post)\n";

my $hmc_options = Datamunge::HMCOptions->new();
$hmc_options->swig_num_warmup_set(1000);
$hmc_options->swig_num_samples_set(4000);
$hmc_options->swig_num_leapfrog_steps_set(15);
$hmc_options->swig_initial_step_size_set(0.3);
my $result = Datamunge::HMC->new($hmc_options)->sample($conjugate_model, [0.0]);
my $m = mean_of($result->swig_samples_get(), 0);
my $s = sd_of($result->swig_samples_get(), 0, $m);
print "HMC:  mu ~ N($m, $s^2), accept rate = " . $result->swig_accept_rate_get() . ", step size = " . $result->swig_final_step_size_get() . "\n";

my $nuts_options = Datamunge::NUTSOptions->new();
$nuts_options->swig_num_warmup_set(1000);
$nuts_options->swig_num_samples_set(4000);
$nuts_options->swig_initial_step_size_set(0.3);
$result = Datamunge::NUTS->new($nuts_options)->sample($conjugate_model, [0.0]);
$m = mean_of($result->swig_samples_get(), 0);
$s = sd_of($result->swig_samples_get(), 0, $m);
print "NUTS: mu ~ N($m, $s^2), accept rate = " . $result->swig_accept_rate_get() . ", step size = " . $result->swig_final_step_size_get() .
      ", divergences = " . $result->swig_num_divergences_get() . "\n";

print "\n=================== Bayesian logistic regression vs. GLM's MLE (iris) ===================\n";
my $iris = Datamunge::DataFrame::iris();
my (@is_virginica, @petal_length);
for (my $i = 0; $i < $iris->nrows(); $i++) {
  my $species = $iris->string_at("Species", $i);
  next unless $species eq "versicolor" || $species eq "virginica";
  push @is_virginica, ($species eq "virginica" ? 1.0 : 0.0);
  push @petal_length, $iris->numeric_at("Petal.Length", $i);
}

my $df = Datamunge::DataFrame->new();
$df->add_numeric_column("Petal.Length", \@petal_length);
$df->add_numeric_column("is_virginica", \@is_virginica);
my $glm = Datamunge::GLM->new($df, "is_virginica ~ Petal.Length", "binomial");
print "GLM MLE:         [" . join(",", @{$glm->coefficients()}) . "]\n";

my $logistic_model = LogisticModel->new(\@petal_length, \@is_virginica);
$coords = Datamunge::DVector->new([0.0, 0.0]);
Datamunge::MAP->new()->optimize($logistic_model, $coords);
print "Bayes MAP:       [" . $coords->get(0) . "," . $coords->get(1) . "] (weak Normal(0, 10) priors)\n";

$nuts_options = Datamunge::NUTSOptions->new();
$nuts_options->swig_num_warmup_set(1000);
$nuts_options->swig_num_samples_set(3000);
$nuts_options->swig_initial_step_size_set(0.05);
$result = Datamunge::NUTS->new($nuts_options)->sample($logistic_model, [0.0, 0.0]);
print "Bayes NUTS mean: [" . mean_of($result->swig_samples_get(), 0) . ", " . mean_of($result->swig_samples_get(), 1) .
      "] (posterior mean, accept rate = " . $result->swig_accept_rate_get() . ")\n";
print "(Petal.Length nearly separates these two species, so the unregularized MLE inflates toward the\n" .
      " separating boundary; the weak Normal(0, 10) prior visibly pulls the Bayesian estimate back --\n" .
      " a real, expected difference, not a bug.)\n";
