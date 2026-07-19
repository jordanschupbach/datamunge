use strict;
use warnings;

use Datamunge;

sub gauss { sqrt(-2.0 * log(rand())) * cos(2.0 * 3.14159265358979 * rand()) }

sub dvector_to_array {
  my ($dv) = @_;
  my @out;
  for (my $i = 0; $i < $dv->size(); $i++) { push @out, $dv->get($i); }
  return \@out;
}

print "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n";
srand(42);
my @y;
my $level = 100.0;
my $prev_shock = 0.0;
for (1 .. 150) {
  my $shock = gauss();
  $level += 0.3 + 0.4 * $prev_shock + $shock;  # integrated ARMA(0,1) increments with drift
  push @y, $level;
  $prev_shock = $shock;
}

my $options = Datamunge::ARIMAOptions->new();
$options->swig_p_set(1);
$options->swig_d_set(1);
$options->swig_q_set(1);
$options->swig_de_population_size_set(80);
$options->swig_de_max_generations_set(400);
my $model = Datamunge::ARIMA->new(\@y, $options);

print "AR coefficient: " . $model->ar_coefficients()->get(0) . "\n";
print "MA coefficient: " . $model->ma_coefficients()->get(0) . "\n";
print "sigma^2: " . $model->sigma2() . ", AIC: " . $model->aic() . ", BIC: " . $model->bic() . "\n";

my $pair = $model->forecast_with_intervals(6);
my $point = dvector_to_array($pair->swig_first_get());
my $se = dvector_to_array($pair->swig_second_get());
print "6-step forecast: " . join(", ", @$point) . "\n";
print "forecast std. errors: " . join(", ", @$se) . "\n";

print "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n";
srand(7);
my @s;
my $prev = 0.0;
for (my $i = 0; $i < 120; $i++) {
  $prev = 0.5 * $prev + gauss();
  push @s, 20.0 + 0.2 * $i + 5.0 * sin(2.0 * 3.14159265358979 * $i / 12.0) + $prev;
}

my $options2 = Datamunge::ARIMAOptions->new();
$options2->swig_p_set(1);
$options2->swig_seasonal_p_set(1);
$options2->swig_seasonal_d_set(1);
$options2->swig_seasonal_period_set(12);
$options2->swig_de_population_size_set(100);
$options2->swig_de_max_generations_set(500);
my $model2 = Datamunge::ARIMA->new(\@s, $options2);
print "AR coefficient: " . $model2->ar_coefficients()->get(0) . ", seasonal AR coefficient: " . $model2->seasonal_ar_coefficients()->get(0) . "\n";
print "12-step forecast: " . join(", ", @{$model2->forecast(12)}) . "\n";

print "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n";
my @seasonal_shape = (0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6);
srand(11);
my @y2;
for (my $i = 0; $i < 48; $i++) {
  my $lvl = 100.0 + 2.0 * $i;
  push @y2, $lvl * $seasonal_shape[$i % 12] + gauss() * 3.0;
}

my $es_options = Datamunge::ExponentialSmoothingOptions->new();
$es_options->swig_trend_set($Datamunge::TrendType_Additive);
$es_options->swig_seasonal_set($Datamunge::SeasonalType_Multiplicative);
$es_options->swig_seasonal_period_set(12);
$es_options->swig_de_population_size_set(60);
$es_options->swig_de_max_generations_set(300);
my $es_model = Datamunge::ExponentialSmoothing->new(\@y2, $es_options);
print "alpha=" . $es_model->alpha() . " beta=" . $es_model->beta() . " gamma=" . $es_model->gamma() . "\n";
print "sigma^2: " . $es_model->sigma2() . ", AIC: " . $es_model->aic() . "\n";
print "12-month forecast: " . join(", ", @{$es_model->forecast(12)}) . "\n";

print "\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n";
my @flat = (50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7);

my $ses = Datamunge::ExponentialSmoothing->new(\@flat, Datamunge::ExponentialSmoothingOptions->new());
print "SES alpha= " . $ses->alpha() . "\n";
print "SES 5-step forecast: " . join(", ", @{$ses->forecast(5)}) . "\n";

my $holt_options = Datamunge::ExponentialSmoothingOptions->new();
$holt_options->swig_trend_set($Datamunge::TrendType_Additive);
my $holt = Datamunge::ExponentialSmoothing->new(\@flat, $holt_options);
print "Holt alpha=" . $holt->alpha() . " beta=" . $holt->beta() . "\n";
print "Holt 5-step forecast: " . join(", ", @{$holt->forecast(5)}) . "\n";
