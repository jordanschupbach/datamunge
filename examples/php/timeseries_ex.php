<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function randn() {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    return sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
}

function cell_join($v) {
    $parts = array();
    for ($i = 0; $i < $v->size(); $i++) $parts[] = $v->get($i);
    return implode(", ", $parts);
}

print("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n");
$y = array();
$level = 100.0;
$prev_shock = 0.0;
for ($i = 0; $i < 150; $i++) {
    $shock = randn();
    $level = $level + 0.3 + 0.4 * $prev_shock + $shock;
    $y[] = $level;
    $prev_shock = $shock;
}

$options = new ARIMAOptions();
$options->p = 1;
$options->d = 1;
$options->q = 1;
$options->de_population_size = 80;
$options->de_max_generations = 400;
$model = new ARIMA(dvector($y), $options);

$ar = $model->ar_coefficients();
$ma = $model->ma_coefficients();
print("AR coefficient: " . $ar->get(0) . "\n");
print("MA coefficient: " . $ma->get(0) . "\n");
print("sigma^2: " . $model->sigma2() . ", AIC: " . $model->aic() . ", BIC: " . $model->bic() . "\n");

$pair = $model->forecast_with_intervals(6);
print("6-step forecast: " . cell_join($pair->first) . "\n");
print("forecast std. errors: " . cell_join($pair->second) . "\n");

print("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n");
$s = array();
$prev = 0.0;
for ($i = 0; $i < 120; $i++) {
    $prev = 0.5 * $prev + randn();
    $s[] = 20.0 + 0.2 * $i + 5.0 * sin((2.0 * M_PI * $i) / 12.0) + $prev;
}

$options2 = new ARIMAOptions();
$options2->p = 1;
$options2->seasonal_p = 1;
$options2->seasonal_d = 1;
$options2->seasonal_period = 12;
$options2->de_population_size = 100;
$options2->de_max_generations = 500;
$model2 = new ARIMA(dvector($s), $options2);
$ar2 = $model2->ar_coefficients();
$sar2 = $model2->seasonal_ar_coefficients();
print("AR coefficient: " . $ar2->get(0) . ", seasonal AR coefficient: " . $sar2->get(0) . "\n");
print("12-step forecast: " . cell_join($model2->forecast(12)) . "\n");

print("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n");
$seasonal_shape = array(0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6);
$y2 = array();
for ($i = 0; $i < 48; $i++) {
    $lvl = 100.0 + 2.0 * $i;
    $y2[] = $lvl * $seasonal_shape[$i % 12] + randn() * 3.0;
}

$es_options = new ExponentialSmoothingOptions();
$es_options->trend = datamunge::TrendType_Additive;
$es_options->seasonal = datamunge::SeasonalType_Multiplicative;
$es_options->seasonal_period = 12;
$es_options->de_population_size = 60;
$es_options->de_max_generations = 300;
$es_model = new ExponentialSmoothing(dvector($y2), $es_options);
print("alpha=" . $es_model->alpha() . " beta=" . $es_model->beta() . " gamma=" . $es_model->gamma() . "\n");
print("sigma^2: " . $es_model->sigma2() . ", AIC: " . $es_model->aic() . "\n");
print("12-month forecast: " . cell_join($es_model->forecast(12)) . "\n");

print("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n");
$flat = array(50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7);

$ses = new ExponentialSmoothing(dvector($flat), new ExponentialSmoothingOptions());
print("SES alpha= " . $ses->alpha() . "\n");
print("SES 5-step forecast: " . cell_join($ses->forecast(5)) . "\n");

$holt_options = new ExponentialSmoothingOptions();
$holt_options->trend = datamunge::TrendType_Additive;
$holt = new ExponentialSmoothing(dvector($flat), $holt_options);
print("Holt alpha=" . $holt->alpha() . " beta=" . $holt->beta() . "\n");
print("Holt 5-step forecast: " . cell_join($holt->forecast(5)) . "\n");

?>
