use strict;
use warnings;

use Datamunge;

print "=================== Random intercept on a real dataset (penguins) ===================\n";
my $penguins = Datamunge::DataFrame::penguins();
my $species_model = Datamunge::LMM->new($penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
$species_model->print_summary();

print "\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n";
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
srand(2024);
sub gauss { sqrt(-2.0 * log(rand())) * cos(2.0 * 3.14159265358979 * rand()) }

my $n_schools = 30;
my @school_intercept = map { gauss() * 6.0 } (1 .. $n_schools);
my @school_slope = map { gauss() * 1.2 } (1 .. $n_schools);

my $true_intercept = 60.0;
my $true_slope = 3.0;
my (@school, @study_hours, @score);
for (my $s = 0; $s < $n_schools; $s++) {
  my $n_students = 15 + int(rand(21));
  for (my $j = 0; $j < $n_students; $j++) {
    my $hours = rand() * 10.0;
    my $noise = gauss() * 4.0;
    my $s_val = $true_intercept + $school_intercept[$s] + ($true_slope + $school_slope[$s]) * $hours + $noise;
    push @school, $s * 1.0;
    push @study_hours, $hours;
    push @score, $s_val;
  }
}

my $df = Datamunge::DataFrame->new();
$df->add_numeric_column("school", \@school);
$df->add_numeric_column("study_hours", \@study_hours);
$df->add_numeric_column("score", \@score);

my $model = Datamunge::LMM->new($df, "score ~ study_hours + (1 + study_hours | school)");
$model->print_summary();

print "\nTrue generating values: intercept=$true_intercept, slope=$true_slope, " .
      "random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n";

print "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n";
my $group_labels = $model->group_labels();
for my $idx (0, 1, 2) {
  my $re = $model->random_effects_for_group($idx);
  print "school " . $group_labels->[$idx] . ": intercept shift=" . $re->[0] . ", slope shift=" . $re->[1] . "\n";
}

print "\n--- Prediction: population-level vs. school-adjusted ---\n";
my $newdata_population = Datamunge::DataFrame->new();
$newdata_population->add_numeric_column("study_hours", [5.0]);
my $newdata_school0 = Datamunge::DataFrame->new();
$newdata_school0->add_numeric_column("study_hours", [5.0]);
$newdata_school0->add_numeric_column("school", [0.0]);
print "5 study hours, unseen school:      " . $model->predict($newdata_population)->[0] . " (fixed effects only)\n";
print "5 study hours, school 0 (known):    " . $model->predict($newdata_school0)->[0] . " (fixed effects + school 0's BLUP)\n";
