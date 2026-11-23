# CMD: just TARGET=gaussian_process_regression_iris_ex run-perl
# Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $formula = "Petal.Length ~ Petal.Width";

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n";
print "formula: $formula\n\n";

# Both the length scale and the noise ratio are auto-selected by maximizing the exact log
# marginal likelihood.
my $model = Datamunge::GaussianProcessRegression->new($iris, $formula);
$model->print_summary();

$model->plot_fit($iris)->save("gpr_iris_fit.svg");
$model->plot_length_scale_profile()->save("gpr_iris_length_scale_profile.svg");
print "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n";

# Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
# interval at every point -- including far outside the training data, where it should widen
# substantially as the model's uncertainty grows.
my $query = Datamunge::DataFrame->new();
$query->add_numeric_column("Petal.Width", [0.2, 1.3, 2.5, 10.0]);
my $detail = $model->predict_frame($query, "confidence");
print "\nPredictions with 95% confidence intervals:\n";
print "$detail\n";
print "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n";
