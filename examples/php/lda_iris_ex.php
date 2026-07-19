<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n\n");

$model = new LDA($iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
$model->print_summary();

print("\nConfusion matrix (rows = actual, cols = predicted):\n");
print($model->confusion_matrix()->to_string() . "\n");

$newdata = new DataFrame();
$newdata->add_numeric_column("Sepal.Length", dvector(array(5.1, 6.0, 6.5, 6.2)));
$newdata->add_numeric_column("Sepal.Width", dvector(array(3.5, 2.7, 3.0, 2.8)));
$newdata->add_numeric_column("Petal.Length", dvector(array(1.4, 4.5, 5.5, 4.8)));
$newdata->add_numeric_column("Petal.Width", dvector(array(0.2, 1.5, 2.0, 1.8)));

print("\nPredictions for new flowers:\n");
print($model->predict_frame($newdata)->to_string() . "\n");

$model->save_discriminant_plot("lda_iris_discriminants.svg");
print("\nSaved discriminant plot as lda_iris_discriminants.svg\n");

?>
