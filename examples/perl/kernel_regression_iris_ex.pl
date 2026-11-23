# CMD: just TARGET=kernel_regression_iris_ex run-perl
# Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $formula = "Petal.Length ~ Petal.Width";

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n";
print "formula: $formula\n\n";

my $model = Datamunge::KernelRegression->new($iris, $formula);  # bandwidth auto-selected via leave-one-out CV
$model->print_summary();

$model->plot_fit($iris)->save("kernel_regression_iris_fit.svg");
$model->plot_cv_curve()->save("kernel_regression_iris_cv.svg");
print "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n";

# A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
# bias-variance story. Datamunge::KernelRegression->new($data, $formula, $kernel, $bandwidth, $n_bandwidth, $standardize)
my $small = Datamunge::KernelRegression->new($iris, $formula, "gaussian", 0.05);
print "\nbandwidth=0.05 (too small): LOO R-squared=", fmt($small->r_squared()), "  LOO RMSE=", fmt($small->rmse()), "\n";
$small->plot_fit($iris)->save("kernel_regression_iris_fit_small_bandwidth.svg");

my $large = Datamunge::KernelRegression->new($iris, $formula, "gaussian", 5.0);
print "bandwidth=5.0 (too large):  LOO R-squared=", fmt($large->r_squared()), "  LOO RMSE=", fmt($large->rmse()), "\n";
$large->plot_fit($iris)->save("kernel_regression_iris_fit_large_bandwidth.svg");

print "bandwidth=", fmt($model->bandwidth()), " (CV-selected): LOO R-squared=", fmt($model->r_squared()), "  LOO RMSE=", fmt($model->rmse()), "\n";
print "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n";

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
