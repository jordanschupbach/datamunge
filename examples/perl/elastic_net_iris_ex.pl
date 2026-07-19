use strict;
use warnings;

use Datamunge;

my $FORMULA = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

my $iris = Datamunge::DataFrame::iris();
print "iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n";
print "formula: $FORMULA\n\n";

print "=================== Ridge ===================\n";
my $ridge = Datamunge::Ridge->new($iris, $FORMULA);
$ridge->print_summary();

print "\n=================== Lasso ===================\n";
my $lasso = Datamunge::Lasso->new($iris, $FORMULA);
$lasso->print_summary();

print "\n================= Elastic Net =================\n";
# ElasticNet(data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
my $elastic = Datamunge::ElasticNet->new($iris, $FORMULA, 0.5);
$elastic->print_summary();

print "\nSaved figures showing how each model's coefficients respond to the " .
      "regularization strength, and the cross-validation curve used to pick it:\n";

$ridge->plot_coefficient_path()->save("elastic_net_ridge_path.svg");
$ridge->plot_cv_curve()->save("elastic_net_ridge_cv.svg");
print "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg\n";

$lasso->plot_coefficient_path()->save("elastic_net_lasso_path.svg");
$lasso->plot_cv_curve()->save("elastic_net_lasso_cv.svg");
print "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg\n";

$elastic->plot_coefficient_path()->save("elastic_net_elasticnet_path.svg");
$elastic->plot_cv_curve()->save("elastic_net_elasticnet_cv.svg");
print "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg\n";
