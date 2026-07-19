<?php

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n");

$model = new DecisionTreeClassifier($iris, "Species ~ Petal.Length + Petal.Width");
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

$model->plot_classification($iris, "Petal.Length", "Petal.Width")->save("tree_iris_classification.svg");
$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("tree_iris_decision_regions.svg");
print("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n");

$shallow = new DecisionTreeClassifier($iris, "Species ~ Petal.Length + Petal.Width", 2);
print("\nDepth-2 tree training accuracy: " . ($shallow->training_accuracy() * 100.0) . "% (" . $shallow->leaf_count() . " leaves)\n");
$shallow->plot_decision_regions("Petal.Length", "Petal.Width")->save("tree_iris_decision_regions_depth2.svg");
print("Saved tree_iris_decision_regions_depth2.svg\n");

?>
