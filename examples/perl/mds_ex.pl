# CMD: just TARGET=mds_ex run-perl
# Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
my @features = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");
my @species = map { $iris->string_at("Species", $_) } 0 .. $iris->nrows() - 1;

print "=================== Classical MDS on iris (euclidean) ===================\n";
my $mds = Datamunge::Mds->new($iris, \@features, 2, "euclidean");
$mds->print_summary();

print "\nEmbedding as a DataFrame:\n", $mds->embedding_frame(), "\n";

$mds->plot_embedding_grouped(\@species)->save_svg("mds_iris_embedding.svg");
print "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n";

print "\n=================== Classical MDS on iris (manhattan) ===================\n";
my $manhattan_mds = Datamunge::Mds->new($iris, \@features, 2, "manhattan");
printf "Goodness of fit: euclidean=%.4f, manhattan=%.4f\n", $mds->goodness_of_fit(), $manhattan_mds->goodness_of_fit();
