# CMD: just TARGET=naive_bayes_penguins_ex run-perl
# Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

# Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
# when fitting, matching every other formula-based model in this library.
my $penguins = Datamunge::Data::penguins();
print "penguins: ", $penguins->nrows(), " rows x ", $penguins->ncols(), " cols\n\n";

# A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
# predictors in one formula -- each modeled independently given the class, per the naive
# Bayes assumption.
my $model = Datamunge::NaiveBayesClassifier->new($penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
$model->print_summary();

print "\nConfusion matrix (rows = actual, cols = predicted):\n", $model->confusion_matrix(), "\n";

print "\nMisclassified rows:\n";
my $predictions = $model->predict($penguins);
my $misclassified = 0;
my @columns = ("species", "bill_length_mm", "bill_depth_mm", "island", "sex");
for my $i (0 .. $penguins->nrows() - 1) {
    next if grep { $penguins->is_null($_, $i) } @columns;
    my ($actual, $predicted) = ($penguins->string_at("species", $i), $predictions->[$i]);
    next if $predicted eq $actual;
    $misclassified++;
    print "  row $i: bill_length=", fmt($penguins->numeric_at("bill_length_mm", $i)), " bill_depth=", fmt($penguins->numeric_at("bill_depth_mm", $i)), " ",
        "island=", $penguins->string_at("island", $i), " sex=", $penguins->string_at("sex", $i), "  actual=$actual  predicted=$predicted\n";
}
print "$misclassified misclassified (of ", $penguins->nrows(), " rows, some incomplete)\n";

# plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
# two-predictor model (bill measurements alone) just for visualization.
my $bill_only = Datamunge::NaiveBayesClassifier->new($penguins, "species ~ bill_length_mm + bill_depth_mm");
print "\nbill-measurements-only model training accuracy: ", fmt(100 * $bill_only->training_accuracy()), "%\n";
$bill_only->plot_classification($penguins, "bill_length_mm", "bill_depth_mm")->save("naive_bayes_penguins_classification.svg");
$bill_only->plot_decision_regions("bill_length_mm", "bill_depth_mm")->save("naive_bayes_penguins_decision_regions.svg");
print "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n";

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
