use strict;
use warnings;

use Datamunge;

sub report_purity {
  my ($cluster_labels, $species, $n_clusters) = @_;
  my @votes = map { {} } (1 .. $n_clusters);
  for (my $i = 0; $i < scalar(@$cluster_labels); $i++) {
    my $label = $cluster_labels->[$i];
    next if $label < 0;
    $votes[$label]{$species->[$i]}++;
  }
  for (my $c = 0; $c < $n_clusters; $c++) {
    my $total = 0;
    $total += $_ for values %{$votes[$c]};
    next if $total == 0;
    my ($best_species) = sort { $votes[$c]{$b} <=> $votes[$c]{$a} } keys %{$votes[$c]};
    my $best = $votes[$c]{$best_species};
    print "  cluster $c: $total points, majority $best_species ($best/$total)\n";
  }
}

my $iris = Datamunge::DataFrame::iris();
my @species = map { $iris->string_at("Species", $_) } (0 .. $iris->nrows() - 1);
my @features = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");

print "=================== K-means (k=3) on iris ===================\n";
my $kmeans = Datamunge::KMeans->new($iris, \@features, 3);
print "Inertia: " . $kmeans->inertia() . ", iterations (best run): " . $kmeans->iterations_used() . "\n";
report_purity($kmeans->labels(), \@species, 3);

print "\n=================== Agglomerative clustering on iris ===================\n";
for my $linkage ("ward", "complete", "average") {
  my $model = Datamunge::AgglomerativeClustering->new($iris, \@features, 3, $linkage);
  print "--- linkage=$linkage ---\n";
  report_purity($model->labels(), \@species, 3);
}

print "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n";
my $ward_model = Datamunge::AgglomerativeClustering->new($iris, \@features, 3, "ward");
report_purity($ward_model->cut(2), \@species, 2);

print "\n=================== DBSCAN on iris ===================\n";
my $dbscan = Datamunge::DBSCAN->new($iris, \@features, 0.6, 5);
print "Clusters found: " . $dbscan->n_clusters() . ", noise points: " . $dbscan->n_noise() . " (of " . $dbscan->observations() . ")\n";
report_purity($dbscan->labels(), \@species, $dbscan->n_clusters());
