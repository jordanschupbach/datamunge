<?php

$penguins = DataFrame::penguins();
print("penguins: " . $penguins->nrows() . " rows x " . $penguins->ncols() . " cols\n\n");

$model = new NaiveBayesClassifier($penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
$model->print_summary();

print("\nConfusion matrix (rows = actual, cols = predicted):\n");
print($model->confusion_matrix()->to_string() . "\n");

print("\nMisclassified rows:\n");
$predictions = $model->predict($penguins);
$misclassified = 0;
$n = $penguins->nrows();
for ($i = 0; $i < $n; $i++) {
    if (!$penguins->is_null("species", $i) && !$penguins->is_null("bill_length_mm", $i) &&
        !$penguins->is_null("bill_depth_mm", $i) && !$penguins->is_null("island", $i) && !$penguins->is_null("sex", $i)) {
        $actual = $penguins->string_at("species", $i);
        $pred = $predictions->get($i);
        if ($pred !== $actual) {
            $misclassified += 1;
            print("  row $i: bill_length=" . $penguins->numeric_at("bill_length_mm", $i) . " bill_depth=" . $penguins->numeric_at("bill_depth_mm", $i) . " island=" . $penguins->string_at("island", $i) . " sex=" . $penguins->string_at("sex", $i) . "  actual=$actual  predicted=$pred\n");
        }
    }
}
print("$misclassified misclassified (of $n rows, some incomplete)\n");

$bill_only = new NaiveBayesClassifier($penguins, "species ~ bill_length_mm + bill_depth_mm");
print("\nbill-measurements-only model training accuracy: " . ($bill_only->training_accuracy() * 100.0) . "%\n");
$bill_only->plot_classification($penguins, "bill_length_mm", "bill_depth_mm")->save("naive_bayes_penguins_classification.svg");
$bill_only->plot_decision_regions("bill_length_mm", "bill_depth_mm")->save("naive_bayes_penguins_decision_regions.svg");
print("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n");

?>
