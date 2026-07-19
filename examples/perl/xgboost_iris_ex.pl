use strict;
use warnings;

use Datamunge;

my $iris = Datamunge::DataFrame::iris();
print "iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n";

my $model = Datamunge::XGBoostClassifier->new($iris, "Species ~ Petal.Length + Petal.Width");
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

$model->plot_training_deviance()->save("xgboost_iris_training_deviance.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("xgboost_iris_decision_regions.svg");
print "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n";

# XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
# grows the same deep trees but keeps every leaf weight small, producing a much smoother
# decision boundary than the lightly-regularized default.
# XGBoostClassifier(data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
#                    min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
my $heavy = Datamunge::XGBoostClassifier->new($iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0);
my $model_dev = $model->training_deviance();
my $heavy_dev = $heavy->training_deviance();
print "\nlambda=1 (default):   training accuracy=" . ($model->training_accuracy() * 100.0) .
      "%  deviance=" . $model_dev->[scalar(@$model_dev) - 1] . "\n";
print "lambda=50 (heavy L2): training accuracy=" . ($heavy->training_accuracy() * 100.0) .
      "%  deviance=" . $heavy_dev->[scalar(@$heavy_dev) - 1] . "\n";
$heavy->plot_decision_regions("Petal.Length", "Petal.Width")->save("xgboost_iris_decision_regions_heavy_lambda.svg");
print "Saved xgboost_iris_decision_regions_heavy_lambda.svg\n";
