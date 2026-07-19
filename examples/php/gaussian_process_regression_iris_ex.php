<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

$FORMULA = "Petal.Length ~ Petal.Width";

$iris = DataFrame::iris();
print("iris: " . $iris->nrows() . " rows x " . $iris->ncols() . " cols\n");
print("formula: $FORMULA\n\n");

$model = new GaussianProcessRegression($iris, $FORMULA);
$model->print_summary();

$model->plot_fit($iris)->save("gpr_iris_fit.svg");
$model->plot_length_scale_profile()->save("gpr_iris_length_scale_profile.svg");
print("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n");

$query = new DataFrame();
$query->add_numeric_column("Petal.Width", dvector(array(0.2, 1.3, 2.5, 10.0)));
$detail = $model->predict_frame($query, "confidence");
print("\nPredictions with 95% confidence intervals:\n");
print($detail->to_string() . "\n");
print("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)\n");

?>
