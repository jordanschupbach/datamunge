# CMD: just TARGET=glm_iris_ex run-perl
# Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

use List::Util qw(min max sum);
use POSIX ();

my $iris = Datamunge::Data::iris();
my @rows = 0 .. $iris->nrows() - 1;

# Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
# the other two on these predictors, which sends logistic regression's coefficients toward
# +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
# binary split for a demo.
my @kept = grep { my $s = $iris->string_at("Species", $_); $s eq "versicolor" || $s eq "virginica" } @rows;
my @is_virginica = map { $iris->string_at("Species", $_) eq "virginica" ? 1.0 : 0.0 } @kept;
my @petal_length = map { $iris->numeric_at("Petal.Length", $_) } @kept;
my @petal_width = map { $iris->numeric_at("Petal.Width", $_) } @kept;

my $sub = Datamunge::DataFrame->new();
$sub->add_numeric_column("Petal.Length", \@petal_length);
$sub->add_numeric_column("Petal.Width", \@petal_width);
$sub->add_numeric_column("is_virginica", \@is_virginica);

print "=================== Logistic regression (binomial, logit link) ===================\n";
my $logit = Datamunge::Glm->new($sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
$logit->print_summary();

my $fitted = $logit->fitted_values();
my $n = @is_virginica;
my $correct = grep { ($fitted->[$_] >= 0.5) == ($is_virginica[$_] >= 0.5) } 0 .. $n - 1;
print "\nResubstitution accuracy at 0.5 threshold: ", fmt(100.0 * $correct / $n), "%\n";

$logit->save_diagnostic_plots("glm_logistic_iris");
print "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n";

# Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
# classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
{
    my $grid_n = 100;
    my $width_mean = sum(@petal_width) / @petal_width;
    my ($pl_min, $pl_max) = (min(@petal_length) - 0.3, max(@petal_length) + 0.3);
    my $grid = Datamunge::DataFrame->new();
    $grid->add_numeric_column("Petal.Length", [map { $pl_min + ($pl_max - $pl_min) * $_ / ($grid_n - 1) } 0 .. $grid_n - 1]);
    $grid->add_numeric_column("Petal.Width", [($width_mean) x $grid_n]);
    my $curve_frame = $logit->predict_frame($grid, "confidence");
    print "\nPredicted-probability curve (first 5 rows):\n";
    print $curve_frame->to_string(5), "\n";
}

# Poisson regression, for contrast: same IRLS engine, different family/link.
print "\n=================== Poisson regression (log link) ===================\n";
my $count_data = Datamunge::DataFrame->new();
$count_data->add_numeric_column("Sepal.Width", [map { $iris->numeric_at("Sepal.Width", $_) } @rows]);
$count_data->add_numeric_column("Petal.Length", [map { $iris->numeric_at("Petal.Length", $_) } @rows]);
$count_data->add_numeric_column("count", [map { POSIX::floor($iris->numeric_at("Sepal.Length", $_) + 0.5) } @rows]);
Datamunge::Glm->new($count_data, "count ~ Sepal.Width + Petal.Length", "poisson")->print_summary();

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
