# CMD: just TARGET=clustering_ex run-perl
# Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
my @species = map { $iris->string_at("Species", $_) } 0 .. $iris->nrows() - 1;
my @features = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");

# For each cluster, print its size and its majority species (negative labels are noise).
sub report_purity {
    my ($labels, $n_clusters) = @_;
    my @votes = map { {} } 1 .. $n_clusters;
    my @order = map { [] } 1 .. $n_clusters;    # species in first-seen order, for ties
    for my $i (0 .. $#$labels) {
        my ($label, $sp) = ($labels->[$i], $species[$i]);
        next if $label < 0;
        push @{ $order[$label] }, $sp unless $votes[$label]{$sp}++;
    }
    for my $c (0 .. $n_clusters - 1) {
        my $total = 0;
        $total += $_ for values %{ $votes[$c] };
        next if $total == 0;
        my ($best_species, $best) = ('', -1);
        for my $sp (@{ $order[$c] }) {
            ($best_species, $best) = ($sp, $votes[$c]{$sp}) if $votes[$c]{$sp} > $best;
        }
        print "  cluster $c: $total points, majority $best_species ($best/$total)\n";
    }
}

print "=================== K-means (k=3) on iris ===================\n";
my $kmeans = Datamunge::KMeans->new($iris, \@features, 3);
print "Inertia: ", fmt($kmeans->inertia()), ", iterations (best run): ", $kmeans->iterations_used(), "\n";
report_purity($kmeans->labels(), 3);

print "\n=================== Agglomerative clustering on iris ===================\n";
for my $linkage ("ward", "complete", "average") {
    my $model = Datamunge::AgglomerativeClustering->new($iris, \@features, 3, $linkage);
    print "--- linkage=$linkage ---\n";
    report_purity($model->labels(), 3);
}

print "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n";
my $ward_model = Datamunge::AgglomerativeClustering->new($iris, \@features, 3, "ward");
report_purity($ward_model->cut(2), 2);

print "\n=================== DBSCAN on iris ===================\n";
my $dbscan = Datamunge::Dbscan->new($iris, \@features, 0.6, 5);
print "Clusters found: ", $dbscan->n_clusters(), ", noise points: ", $dbscan->n_noise(), " (of ", $dbscan->observations(), ")\n";
report_purity($dbscan->labels(), $dbscan->n_clusters());

# Formats a number like C#'s double.ToString(): shortest round-trip digits, E notation outside [1e-4, 1e15).
sub fmt {
    my ($x) = @_;
    return 'NaN' if $x != $x;
    return $x > 0 ? 'Infinity' : '-Infinity' if $x == 9**9**9 || $x == -9**9**9;
    return '0' if $x == 0;
    my $s;
    for my $p (0 .. 16) { $s = sprintf('%.*e', $p, $x); last if $s == $x }
    my ($m, $e) = $s =~ /^(-?[\d.]+)e([-+]\d+)$/;
    $m =~ s/\.?0+$// if $m =~ /\./;
    $e += 0;
    return sprintf('%sE%s%02d', $m, $e < 0 ? '-' : '+', abs $e) if $e < -4 || $e >= 15;
    my $neg = $m =~ s/^-//;
    my $digits = $m =~ s/\.//r;
    my $out;
    if ($e < 0) { $out = '0.' . ('0' x (-$e - 1)) . $digits }
    elsif (length($digits) > $e + 1) { $out = substr($digits, 0, $e + 1) . '.' . substr($digits, $e + 1) }
    else { $out = $digits . ('0' x ($e + 1 - length $digits)) }
    return ($neg ? '-' : '') . $out;
}
