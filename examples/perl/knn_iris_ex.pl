# CMD: just TARGET=knn_iris_ex run-perl
# k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n\n";

my $model = Datamunge::KnnClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n";
print $model->confusion_matrix(), "\n";

print "\nLeave-one-out misclassified rows:\n";

my $predictions = $model->predict($iris);
my $n = $iris->nrows();
my $misclassified = 0;
for my $i (0 .. $n - 1) {
    my ($actual, $predicted) = ($iris->string_at("Species", $i), $predictions->[$i]);
    next if $predicted eq $actual;
    $misclassified++;
    print "  row $i: Petal.Length=", fmt($iris->numeric_at("Petal.Length", $i)), " Petal.Width=", fmt($iris->numeric_at("Petal.Width", $i)),
        "  actual=$actual  predicted=$predicted\n";
}
print "$misclassified of $n misclassified (", fmt(100.0 * $misclassified / $n), "%)\n";

$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k5.svg");
print "\nSaved knn_iris_decision_regions_k5.svg\n";

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). Datamunge::KnnClassifier->new($data, $formula, $k, $metric, $weighted, $standardize)
for my $k (1, 25) {
    my $knn = Datamunge::KnnClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", $k);
    my $file = "knn_iris_decision_regions_k$k.svg";
    printf "\nk=%-2s leave-one-out accuracy: %s%%\n", $k, fmt($knn->training_accuracy() * 100);
    $knn->plot_decision_regions("Petal.Length", "Petal.Width")->save($file);
    print "Saved $file\n";
}

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
