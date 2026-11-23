# CMD: just TARGET=tree_iris_ex run-perl
# Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n\n";

my $model = Datamunge::DecisionTreeClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print "\nConfusion matrix (rows = actual, cols = predicted):\n";
print $model->confusion_matrix(), "\n";

print "\nMisclassified rows:\n";

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

$model->plot_classification($iris, "Petal.Length", "Petal.Width")->save("tree_iris_classification.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("tree_iris_decision_regions.svg");
print "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n";

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
my $shallow = Datamunge::DecisionTreeClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 2);
print "\nDepth-2 tree training accuracy: ", fmt($shallow->training_accuracy() * 100), "% (", $shallow->leaf_count(), " leaves)\n";
$shallow->plot_decision_regions("Petal.Length", "Petal.Width")->save("tree_iris_decision_regions_depth2.svg");
print "Saved tree_iris_decision_regions_depth2.svg\n";

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
