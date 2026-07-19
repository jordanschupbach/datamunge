use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
print "iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n";

my $model = Datamunge::GBMClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
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

$model->plot_training_deviance()->save("gbm_iris_training_deviance.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("gbm_iris_decision_regions.svg");
print "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n";

# A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
# at the training loss, gradually sharpening the decision boundary.
# GBMClassifier(data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
my $few = Datamunge::GBMClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 5);
my $few_dev = $few->training_deviance();
my $model_dev = $model->training_deviance();
print "\n5-round ensemble:   training accuracy=" . ($few->training_accuracy() * 100.0) .
      "%  deviance=" . $few_dev->[scalar(@$few_dev) - 1] . "\n";
print "100-round ensemble: training accuracy=" . ($model->training_accuracy() * 100.0) .
      "%  deviance=" . $model_dev->[scalar(@$model_dev) - 1] . "\n";
$few->plot_decision_regions("Petal.Length", "Petal.Width")->save("gbm_iris_decision_regions_5rounds.svg");
print "Saved gbm_iris_decision_regions_5rounds.svg\n";
