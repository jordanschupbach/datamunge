use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
print "iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n";

my $model = Datamunge::DecisionTreeClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print "\nConfusion matrix (rows = actual, cols = predicted):\n";
print $model->confusion_matrix()->to_string(), "\n";

print "\nMisclassified rows:\n";
my $predictions = $model->predict($iris);
my $misclassified = 0;
for (my $i = 0; $i < $iris->nrows(); $i++) {
  my $actual = $iris->string_at("Species", $i);
  next if $predictions->[$i] eq $actual;
  $misclassified++;
  print "  row $i: Petal.Length=" . $iris->numeric_at("Petal.Length", $i) .
        " Petal.Width=" . $iris->numeric_at("Petal.Width", $i) .
        "  actual=$actual  predicted=" . $predictions->[$i] . "\n";
}
print "$misclassified of " . $iris->nrows() . " misclassified (" . (100.0 * $misclassified / $iris->nrows()) . "%)\n";

$model->plot_classification($iris, "Petal.Length", "Petal.Width")->save("tree_iris_classification.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("tree_iris_decision_regions.svg");
print "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n";

# A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
my $shallow = Datamunge::DecisionTreeClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 2);
print "\nDepth-2 tree training accuracy: " . ($shallow->training_accuracy() * 100.0) . "% (" . $shallow->leaf_count() . " leaves)\n";
$shallow->plot_decision_regions("Petal.Length", "Petal.Width")->save("tree_iris_decision_regions_depth2.svg");
print "Saved tree_iris_decision_regions_depth2.svg\n";
