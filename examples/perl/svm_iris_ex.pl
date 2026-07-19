use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
print "iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n";

print "=== RBF kernel (default) ===\n";
my $rbf_model = Datamunge::SVM->new($iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
$rbf_model->print_summary();
print "\nConfusion matrix (rows = actual, cols = predicted):\n";
print $rbf_model->confusion_matrix()->to_string(), "\n";

print "\n=== Linear kernel, for comparison ===\n";
my $linear_model = Datamunge::SVM->new($iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear");
print "Training accuracy: " . ($linear_model->training_accuracy() * 100.0) . "%\n";
print "Support vectors: " . $linear_model->num_support_vectors() . "\n";

my $newdata = Datamunge::DataFrame->new();
$newdata->add_numeric_column("Sepal.Length", [5.1, 6.0, 6.5, 6.2]);
$newdata->add_numeric_column("Sepal.Width", [3.5, 2.7, 3.0, 2.8]);
$newdata->add_numeric_column("Petal.Length", [1.4, 4.5, 5.5, 4.8]);
$newdata->add_numeric_column("Petal.Width", [0.2, 1.5, 2.0, 1.8]);

print "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n";
print $rbf_model->predict_frame($newdata)->to_string(), "\n";
