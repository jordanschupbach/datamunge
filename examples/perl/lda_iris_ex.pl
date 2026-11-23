# CMD: just TARGET=lda_iris_ex run-perl
# Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
use strict;
use warnings;

use Datamunge qw(:all);

$| = 1;    # flush each print so it interleaves with the C++ library's output

my $iris = Datamunge::Data::iris();
print "iris: ", $iris->nrows(), " rows x ", $iris->ncols(), " cols\n\n";

my $model = Datamunge::Lda->new($iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
$model->print_summary();

print "\nConfusion matrix (rows = actual, cols = predicted):\n", $model->confusion_matrix(), "\n";

my $newdata = Datamunge::DataFrame->new();
$newdata->add_numeric_column("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
$newdata->add_numeric_column("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
$newdata->add_numeric_column("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
$newdata->add_numeric_column("Petal.Width", [0.2, 1.5, 2.0, 1.8]);
print "\nPredictions for new flowers:\n", $model->predict_frame($newdata), "\n";

$model->save_discriminant_plot("lda_iris_discriminants.svg");
print "\nSaved discriminant plot as lda_iris_discriminants.svg\n";
