<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function svector($values) {
    $out = new SVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

$hp = array(110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0);
$wt = array(2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44);
$transmission = array("manual", "manual", "manual", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic", "automatic");
$mpg = array(21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2);

$cars = new DataFrame();
$cars->add_numeric_column("hp", dvector($hp));
$cars->add_numeric_column("wt", dvector($wt));
$cars->add_string_column("transmission", svector($transmission));
$cars->add_numeric_column("mpg", dvector($mpg));

print("Fitting: mpg ~ hp + wt + transmission\n\n");
$model = new LM($cars, "mpg ~ hp + wt + transmission");
$model->print_summary();

print("\nSequential ANOVA:\n");
print($model->anova()->to_string() . "\n");

$newcars = new DataFrame();
$newcars->add_numeric_column("hp", dvector(array(150.0, 90.0)));
$newcars->add_numeric_column("wt", dvector(array(3.0, 2.5)));
$newcars->add_string_column("transmission", svector(array("manual", "automatic")));

$frame = $model->predict_frame($newcars, "confidence");
print("\nPredictions with 95% confidence intervals:\n");
print($frame->to_string() . "\n");

$model->save_diagnostic_plots("lm_ex_diagnostics");
print("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n");

?>
