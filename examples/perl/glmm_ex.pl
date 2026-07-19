use strict;
use warnings;

use Datamunge;

print "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n";
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
my $penguins = Datamunge::DataFrame::penguins();
my (@is_male, @body_mass, @island);
for (my $i = 0; $i < $penguins->nrows(); $i++) {
  next if $penguins->is_null("sex", $i) || $penguins->is_null("body_mass_g", $i) || $penguins->is_null("island", $i);
  push @is_male, ($penguins->string_at("sex", $i) eq "male" ? 1.0 : 0.0);
  push @body_mass, $penguins->numeric_at("body_mass_g", $i);
  push @island, $penguins->string_at("island", $i);
}

my $sex_df = Datamunge::DataFrame->new();
$sex_df->add_numeric_column("is_male", \@is_male);
$sex_df->add_numeric_column("body_mass_g", \@body_mass);
$sex_df->add_string_column("island", \@island);

my $sex_model = Datamunge::GLMM->new($sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial");
$sex_model->print_summary();

print "\n=================== Poisson mixed model on simulated multi-site count data ===================\n";
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
srand(4242);
sub gauss { sqrt(-2.0 * log(rand())) * cos(2.0 * 3.14159265358979 * rand()) }

my $n_stores = 25;
my @store_effect = map { gauss() * 0.4 } (1 .. $n_stores);

my $true_intercept = 2.0;
my $true_slope = 0.3;
my (@store, @promo, @visits);
for (my $s = 0; $s < $n_stores; $s++) {
  my $n_days = 15 + int(rand(11));
  for (my $d = 0; $d < $n_days; $d++) {
    my $promo_intensity = rand() * 3.0;
    my $lam = exp($true_intercept + $store_effect[$s] + $true_slope * $promo_intensity);
    # Knuth's Poisson sampler.
    my $l_thresh = exp(-$lam);
    my $k = 0;
    my $p = 1.0;
    while (1) {
      $k++;
      $p *= rand();
      last if $p <= $l_thresh;
    }
    push @store, $s * 1.0;
    push @promo, $promo_intensity;
    push @visits, ($k - 1) * 1.0;
  }
}

my $df = Datamunge::DataFrame->new();
$df->add_numeric_column("store", \@store);
$df->add_numeric_column("promo", \@promo);
$df->add_numeric_column("visits", \@visits);

my $store_model = Datamunge::GLMM->new($df, "visits ~ promo + (1 | store)", "poisson");
$store_model->print_summary();

print "\nTrue generating values: intercept=$true_intercept, slope=$true_slope, " .
      "random-intercept SD (log scale)=0.4\n";

print "\n--- BLUPs for a few stores ---\n";
my $group_labels = $store_model->group_labels();
for my $idx (0, 1, 2) {
  print "store " . $group_labels->[$idx] . ": intercept shift=" . $store_model->random_effects_for_group($idx)->[0] . "\n";
}

print "\n--- Prediction: population-level vs. store-adjusted ---\n";
my $newdata_population = Datamunge::DataFrame->new();
$newdata_population->add_numeric_column("promo", [1.5]);
my $newdata_store0 = Datamunge::DataFrame->new();
$newdata_store0->add_numeric_column("promo", [1.5]);
$newdata_store0->add_numeric_column("store", [0.0]);
print "promo=1.5, unseen store:   " . $store_model->predict($newdata_population)->[0] . " expected visits (fixed effects only)\n";
print "promo=1.5, store 0 (known): " . $store_model->predict($newdata_store0)->[0] . " expected visits (fixed effects + store 0's BLUP)\n";
