use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
my (@is_virginica, @petal_length, @petal_width);
for (my $i = 0; $i < $iris->nrows(); $i++) {
  my $species = $iris->string_at("Species", $i);
  next unless $species eq "versicolor" || $species eq "virginica";
  push @is_virginica, ($species eq "virginica" ? 1.0 : 0.0);
  push @petal_length, $iris->numeric_at("Petal.Length", $i);
  push @petal_width, $iris->numeric_at("Petal.Width", $i);
}

my $sub = Datamunge::DataFrame->new();
$sub->add_numeric_column("Petal.Length", \@petal_length);
$sub->add_numeric_column("Petal.Width", \@petal_width);
$sub->add_numeric_column("is_virginica", \@is_virginica);

print "=================== Logistic regression (binomial, logit link) ===================\n";
my $logit = Datamunge::GLM->new($sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
$logit->print_summary();

my $fitted = $logit->fitted_values();
my $correct = 0;
for (my $i = 0; $i < scalar(@is_virginica); $i++) {
  $correct++ if (($fitted->[$i] >= 0.5) == ($is_virginica[$i] >= 0.5));
}
print "\nResubstitution accuracy at 0.5 threshold: " . (100.0 * $correct / scalar(@is_virginica)) . "%\n";

$logit->save_diagnostic_plots("glm_logistic_iris");
print "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n";

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
my $width_sum = 0;
$width_sum += $_ for @petal_width;
my $width_mean = $width_sum / scalar(@petal_width);
my $grid_n = 100;
my $pl_min = (sort { $a <=> $b } @petal_length)[0] - 0.3;
my $pl_max = (sort { $b <=> $a } @petal_length)[0] + 0.3;
my @grid_x = map { $pl_min + ($pl_max - $pl_min) * $_ / ($grid_n - 1) } (0 .. $grid_n - 1);
my $grid = Datamunge::DataFrame->new();
$grid->add_numeric_column("Petal.Length", \@grid_x);
$grid->add_numeric_column("Petal.Width", [($width_mean) x $grid_n]);
my $curve_frame = $logit->predict_frame($grid, "confidence");
print "\nPredicted-probability curve (first 5 rows):\n";
print $curve_frame->to_string(5), "\n";

# Poisson regression, for contrast: same IRLS engine, different family/link.
print "\n=================== Poisson regression (log link) ===================\n";
my @count = map { int($iris->numeric_at("Sepal.Length", $_) + 0.5) } (0 .. $iris->nrows() - 1);
my $count_data = Datamunge::DataFrame->new();
my @sepal_width = map { $iris->numeric_at("Sepal.Width", $_) } (0 .. $iris->nrows() - 1);
my @all_petal_length = map { $iris->numeric_at("Petal.Length", $_) } (0 .. $iris->nrows() - 1);
$count_data->add_numeric_column("Sepal.Width", \@sepal_width);
$count_data->add_numeric_column("Petal.Length", \@all_petal_length);
$count_data->add_numeric_column("count", [map { $_ * 1.0 } @count]);

my $poisson = Datamunge::GLM->new($count_data, "count ~ Sepal.Width + Petal.Length", "poisson");
$poisson->print_summary();
