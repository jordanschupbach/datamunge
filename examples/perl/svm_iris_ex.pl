# CMD: just TARGET=svm_iris_ex run-perl
# Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width";
my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n\n";

print "=== RBF kernel (default) ===\n";
my $rbf_model = Datamunge::Svm->new($iris, $formula);
$rbf_model->print_summary();
print "\nConfusion matrix (rows = actual, cols = predicted):\n", $rbf_model->confusion_matrix(), "\n";

print "\n=== Linear kernel, for comparison ===\n";
my $linear_model = Datamunge::Svm->new($iris, $formula, "linear");
print "Training accuracy: ", fmt($linear_model->training_accuracy() * 100), "%\n";
print "Support vectors: ", $linear_model->num_support_vectors(), "\n";

my $newdata = Datamunge::DataFrame->new();
$newdata->add_numeric_column("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
$newdata->add_numeric_column("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
$newdata->add_numeric_column("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
$newdata->add_numeric_column("Petal.Width", [0.2, 1.5, 2.0, 1.8]);
print "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n", $rbf_model->predict_frame($newdata), "\n";

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
