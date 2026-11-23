# CMD: just TARGET=elastic_net_iris_ex run-perl
# Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n";
print "formula: $formula\n\n";

print "=================== Ridge ===================\n";
my $ridge = Datamunge::Ridge->new($iris, $formula);
$ridge->print_summary();

print "\n=================== Lasso ===================\n";
my $lasso = Datamunge::Lasso->new($iris, $formula);
$lasso->print_summary();

print "\n================= Elastic Net =================\n";
# Datamunge::ElasticNet->new($data, $formula, $alpha = 0.5, $lambda, $n_lambda, $cv_folds, $standardize, $seed)
my $elastic = Datamunge::ElasticNet->new($iris, $formula, 0.5);
$elastic->print_summary();

print "\nSaved figures showing how each model's coefficients respond to the regularization strength, "
    . "and the cross-validation curve used to pick it:\n";

my @figures = (
    [$ridge->plot_coefficient_path(), $ridge->plot_cv_curve(), "ridge", "ridge:      "],
    [$lasso->plot_coefficient_path(), $lasso->plot_cv_curve(), "lasso", "lasso:      "],
    [$elastic->plot_coefficient_path(), $elastic->plot_cv_curve(), "elasticnet", "elastic net:"],
);
for my $figure (@figures) {
    my ($path, $cv, $name, $label) = @$figure;
    $path->save("elastic_net_${name}_path.svg");
    $cv->save("elastic_net_${name}_cv.svg");
    print "  $label elastic_net_${name}_path.svg, elastic_net_${name}_cv.svg\n";
}
