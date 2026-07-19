<?php

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n");

$model = new RandomForestClassifier($iris, "Species ~ Petal.Length + Petal.Width");
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

$model->plot_classification($iris, "Petal.Length", "Petal.Width")->save("forest_iris_classification.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("forest_iris_decision_regions.svg");
print("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n");

$small_forest = new RandomForestClassifier($iris, "Species ~ Petal.Length + Petal.Width", 5);
print("\n5-tree forest:   training accuracy=" . ($small_forest->training_accuracy() * 100.0) . "%  OOB accuracy=" . ($small_forest->oob_accuracy() * 100.0) . "%\n");
print("100-tree forest: training accuracy=" . ($model->training_accuracy() * 100.0) . "%  OOB accuracy=" . ($model->oob_accuracy() * 100.0) . "%\n");
$small_forest->plot_decision_regions("Petal.Length", "Petal.Width")->save("forest_iris_decision_regions_5trees.svg");
print("Saved forest_iris_decision_regions_5trees.svg\n");

?>
