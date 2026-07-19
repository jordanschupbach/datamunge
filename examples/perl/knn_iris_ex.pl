use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
print "iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n";

my $model = Datamunge::KNNClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n";
print $model->confusion_matrix()->to_string(), "\n";

print "\nLeave-one-out misclassified rows:\n";
my $fitted = $model->predict($iris);
my $misclassified = 0;
for (my $i = 0; $i < $iris->nrows(); $i++) {
  my $actual = $iris->string_at("Species", $i);
  next if $fitted->[$i] eq $actual;
  $misclassified++;
  print "  row $i: Petal.Length=" . $iris->numeric_at("Petal.Length", $i) .
        " Petal.Width=" . $iris->numeric_at("Petal.Width", $i) .
        "  actual=$actual  predicted=" . $fitted->[$i] . "\n";
}
print "$misclassified of " . $iris->nrows() . " misclassified (" . (100.0 * $misclassified / $iris->nrows()) . "%)\n";

$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k5.svg");
print "\nSaved knn_iris_decision_regions_k5.svg\n";

# k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
# every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
# underfit boundary). KNNClassifier(data, formula, k, metric, weighted, standardize)
my $k1 = Datamunge::KNNClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 1);
print "\nk=1  leave-one-out accuracy: " . ($k1->training_accuracy() * 100.0) . "%\n";
$k1->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k1.svg");
print "Saved knn_iris_decision_regions_k1.svg\n";

my $k25 = Datamunge::KNNClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 25);
print "\nk=25 leave-one-out accuracy: " . ($k25->training_accuracy() * 100.0) . "%\n";
$k25->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k25.svg");
print "Saved knn_iris_decision_regions_k25.svg\n";
