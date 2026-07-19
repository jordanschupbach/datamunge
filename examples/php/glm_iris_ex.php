<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

$iris = DataFrame::iris();

$is_virginica = array();
$petal_length = array();
$petal_width = array();
$n = $iris->nrows();
for ($i = 0; $i < $n; $i++) {
    $species = $iris->string_at("Species", $i);
    if ($species === "versicolor" || $species === "virginica") {
        $is_virginica[] = $species === "virginica" ? 1.0 : 0.0;
        $petal_length[] = $iris->numeric_at("Petal.Length", $i);
        $petal_width[] = $iris->numeric_at("Petal.Width", $i);
    }
}

$sub = new DataFrame();
$sub->add_numeric_column("Petal.Length", dvector($petal_length));
$sub->add_numeric_column("Petal.Width", dvector($petal_width));
$sub->add_numeric_column("is_virginica", dvector($is_virginica));

print("=================== Logistic regression (binomial, logit link) ===================\n");
$logit = new GLM($sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial");
$logit->print_summary();

$fitted = $logit->fitted_values();
$correct = 0;
$nn = count($is_virginica);
for ($i = 0; $i < $nn; $i++) {
    if (($fitted->get($i) >= 0.5) === ($is_virginica[$i] >= 0.5)) $correct += 1;
}
print("\nResubstitution accuracy at 0.5 threshold: " . (100.0 * $correct / $nn) . "%\n");

$logit->save_diagnostic_plots("glm_logistic_iris");
print("\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n");

$width_mean = array_sum($petal_width) / $nn;
$grid_n = 100;
$pl_min = min($petal_length) - 0.3;
$pl_max = max($petal_length) + 0.3;
$grid_x = array();
$grid_w = array();
for ($i = 0; $i < $grid_n; $i++) {
    $grid_x[] = $pl_min + ($pl_max - $pl_min) * $i / ($grid_n - 1);
    $grid_w[] = $width_mean;
}
$grid = new DataFrame();
$grid->add_numeric_column("Petal.Length", dvector($grid_x));
$grid->add_numeric_column("Petal.Width", dvector($grid_w));
$curve_frame = $logit->predict_frame($grid, "confidence");
print("\nPredicted-probability curve (first 5 rows):\n");
print($curve_frame->to_string(5) . "\n");

print("\n=================== Poisson regression (log link) ===================\n");
$count = array();
$sepal_width = array();
$all_petal_length = array();
for ($i = 0; $i < $n; $i++) {
    $count[] = round($iris->numeric_at("Sepal.Length", $i));
    $sepal_width[] = $iris->numeric_at("Sepal.Width", $i);
    $all_petal_length[] = $iris->numeric_at("Petal.Length", $i);
}
$count_data = new DataFrame();
$count_data->add_numeric_column("Sepal.Width", dvector($sepal_width));
$count_data->add_numeric_column("Petal.Length", dvector($all_petal_length));
$count_data->add_numeric_column("count", dvector($count));

$poisson = new GLM($count_data, "count ~ Sepal.Width + Petal.Length", "poisson");
$poisson->print_summary();

?>
