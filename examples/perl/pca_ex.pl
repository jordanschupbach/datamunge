use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
my @features = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");
my @species = map { $iris->string_at("Species", $_) } (0 .. $iris->nrows() - 1);

print "=================== PCA on iris (scaled) ===================\n";
my $pca = Datamunge::PCA->new($iris, \@features, 1, 1);
$pca->print_summary();

print "\nPC1 loadings (which original features drive it):\n";
my $names = $pca->feature_names();
my $loadings = $pca->component_loadings(0);
for (my $i = 0; $i < scalar(@$names); $i++) {
  printf("  %s: %.4f\n", $names->[$i], $loadings->[$i]);
}

print "\nTraining scores as a DataFrame:\n";
print $pca->scores_frame()->to_string(), "\n";

my $scatter = $pca->plot_scores_grouped(\@species);
$scatter->save_svg("pca_iris_scores.svg");
print "\nScores scatter (colored by species) saved as pca_iris_scores.svg\n";

my $scree = $pca->plot_scree();
$scree->save_svg("pca_iris_scree.svg");
print "Scree plot saved as pca_iris_scree.svg\n";

print "\n=================== PCA on iris (unscaled) ===================\n";
my $unscaled = Datamunge::PCA->new($iris, \@features, 1, 0);
my $scaled_ratio = $pca->explained_variance_ratio();
my $unscaled_ratio = $unscaled->explained_variance_ratio();
printf(
  "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance "
    . "dominates the unscaled covariance matrix.\n",
  $unscaled_ratio->[0] * 100, $scaled_ratio->[0] * 100);
