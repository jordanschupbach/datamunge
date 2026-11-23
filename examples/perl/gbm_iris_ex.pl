# CMD: just TARGET=gbm_iris_ex run-perl
# Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n\n";

my $model = Datamunge::GbmClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print "\nConfusion matrix (rows = actual, cols = predicted):\n", $model->confusion_matrix(), "\n";

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

$model->plot_training_deviance()->save("gbm_iris_training_deviance.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("gbm_iris_decision_regions.svg");
print "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n";

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# Datamunge::GbmClassifier->new($data, $formula, $n_trees, $learning_rate, $max_depth, $min_samples_split,
#                               $min_samples_leaf, $subsample, $seed)
my $few = Datamunge::GbmClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 5);
print "\n5-round ensemble:   training accuracy=", fmt(100 * $few->training_accuracy()), "%  deviance=", fmt($few->training_deviance()->[-1]), "\n";
print "100-round ensemble: training accuracy=", fmt(100 * $model->training_accuracy()), "%  deviance=", fmt($model->training_deviance()->[-1]), "\n";
$few->plot_decision_regions("Petal.Length", "Petal.Width")->save("gbm_iris_decision_regions_5rounds.svg");
print "Saved gbm_iris_decision_regions_5rounds.svg\n";

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
