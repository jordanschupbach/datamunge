# CMD: just TARGET=pca_ex run-perl
# PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
my @features = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");

# species labels, for the grouped (colored-by-species) scores plot
my @species = map { $iris->string_at("Species", $_) } 0 .. $iris->nrows() - 1;

print "=================== PCA on iris (scaled) ===================\n";
my $pca = Datamunge::Pca->new($iris, \@features, 1, 1);
$pca->print_summary();

print "\nPC1 loadings (which original features drive it):\n";
my $names = $pca->feature_names();
my $loadings = $pca->component_loadings(0);
printf "  %s: %.4f\n", $names->[$_], $loadings->[$_] for 0 .. $#$names;

print "\nTraining scores as a DataFrame:\n", $pca->scores_frame(), "\n";

$pca->plot_scores_grouped(\@species)->save_svg("pca_iris_scores.svg");
print "\nScores scatter (colored by species) saved as pca_iris_scores.svg\n";

$pca->plot_scree()->save_svg("pca_iris_scree.svg");
print "Scree plot saved as pca_iris_scree.svg\n";

print "\n=================== PCA on iris (unscaled) ===================\n";
my $unscaled = Datamunge::Pca->new($iris, \@features, 1, 0);
printf "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n",
    100 * $unscaled->explained_variance_ratio()->[0], 100 * $pca->explained_variance_ratio()->[0];
