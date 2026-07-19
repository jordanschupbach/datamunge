<?php

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n");

$model = new KNNClassifier($iris, "Species ~ Petal.Length + Petal.Width");
$model->print_summary();

print("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n");
print($model->confusion_matrix()->to_string() . "\n");

print("\nLeave-one-out misclassified rows:\n");
$fitted = $model->predict($iris);
$misclassified = 0;
$n = $iris->nrows();
for ($i = 0; $i < $n; $i++) {
    $actual = $iris->string_at("Species", $i);
    $pred = $fitted->get($i);
    if ($pred !== $actual) {
        $misclassified += 1;
        print("  row $i: Petal.Length=" . $iris->numeric_at("Petal.Length", $i) . " Petal.Width=" . $iris->numeric_at("Petal.Width", $i) . "  actual=$actual  predicted=$pred\n");
    }
}
print("$misclassified of $n misclassified (" . (100.0 * $misclassified / $n) . "%)\n");

$model->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k5.svg");
print("\nSaved knn_iris_decision_regions_k5.svg\n");

$k1 = new KNNClassifier($iris, "Species ~ Petal.Length + Petal.Width", 1);
print("\nk=1  leave-one-out accuracy: " . ($k1->training_accuracy() * 100.0) . "%\n");
$k1->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k1.svg");
print("Saved knn_iris_decision_regions_k1.svg\n");

$k25 = new KNNClassifier($iris, "Species ~ Petal.Length + Petal.Width", 25);
print("\nk=25 leave-one-out accuracy: " . ($k25->training_accuracy() * 100.0) . "%\n");
$k25->plot_decision_regions("Petal.Length", "Petal.Width")->save("knn_iris_decision_regions_k25.svg");
print("Saved knn_iris_decision_regions_k25.svg\n");

?>
