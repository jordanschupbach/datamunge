use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
my @features = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");
my @species = map { $iris->string_at("Species", $_) } (0 .. $iris->nrows() - 1);

print "=================== Classical MDS on iris (euclidean) ===================\n";
my $mds = Datamunge::MDS->new($iris, \@features, 2, "euclidean");
$mds->print_summary();

print "\nEmbedding as a DataFrame:\n";
print $mds->embedding_frame()->to_string(), "\n";

my $scatter = $mds->plot_embedding_grouped(\@species);
$scatter->save_svg("mds_iris_embedding.svg");
print "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n";

print "\n=================== Classical MDS on iris (manhattan) ===================\n";
my $manhattan_mds = Datamunge::MDS->new($iris, \@features, 2, "manhattan");
printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n",
  $mds->goodness_of_fit(), $manhattan_mds->goodness_of_fit());
