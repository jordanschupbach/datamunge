<?php

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n");

$model = new GBMClassifier($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print("\nConfusion matrix (rows = actual, cols = predicted):\n");
print($model->confusion_matrix()->to_string() . "\n");

print("\nMisclassified rows:\n");
$predictions = $model->predict($iris);
$misclassified = 0;
$n = $iris->nrows();
for ($i = 0; $i < $n; $i++) {
    $actual = $iris->string_at("Species", $i);
    $pred = $predictions->get($i);
    if ($pred !== $actual) {
        $misclassified += 1;
        print("  row $i: Petal.Length=" . $iris->numeric_at("Petal.Length", $i) . " Petal.Width=" . $iris->numeric_at("Petal.Width", $i) . "  actual=$actual  predicted=$pred\n");
    }
}
print("$misclassified of $n misclassified (" . (100.0 * $misclassified / $n) . "%)\n");

$model->plot_training_deviance()->save("gbm_iris_training_deviance.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("gbm_iris_decision_regions.svg");
print("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n");

$few = new GBMClassifier($iris, "Species ~ Petal.Length + Petal.Width", 5);
$few_dev = $few->training_deviance();
$model_dev = $model->training_deviance();
print("\n5-round ensemble:   training accuracy=" . ($few->training_accuracy() * 100.0) . "%  deviance=" . $few_dev->get($few_dev->size() - 1) . "\n");
print("100-round ensemble: training accuracy=" . ($model->training_accuracy() * 100.0) . "%  deviance=" . $model_dev->get($model_dev->size() - 1) . "\n");
$few->plot_decision_regions("Petal.Length", "Petal.Width")->save("gbm_iris_decision_regions_5rounds.svg");
print("Saved gbm_iris_decision_regions_5rounds.svg\n");

?>
