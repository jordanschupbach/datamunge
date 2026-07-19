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

function randn() {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    return sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
}

print("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n");
$penguins = DataFrame::penguins();
$is_male = array();
$body_mass = array();
$island = array();
$n = $penguins->nrows();
for ($i = 0; $i < $n; $i++) {
    if (!$penguins->is_null("sex", $i) && !$penguins->is_null("body_mass_g", $i) && !$penguins->is_null("island", $i)) {
        $is_male[] = $penguins->string_at("sex", $i) === "male" ? 1.0 : 0.0;
        $body_mass[] = $penguins->numeric_at("body_mass_g", $i);
        $island[] = $penguins->string_at("island", $i);
    }
}

$sex_df = new DataFrame();
$sex_df->add_numeric_column("is_male", dvector($is_male));
$sex_df->add_numeric_column("body_mass_g", dvector($body_mass));
$sex_df->add_string_column("island", svector($island));

$sex_model = new GLMM($sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial");
$sex_model->print_summary();

print("\n=================== Poisson mixed model on simulated multi-site count data ===================\n");
$n_stores = 25;
$store_effect = array();
for ($i = 0; $i < $n_stores; $i++) $store_effect[] = randn() * 0.4;

$true_intercept = 2.0;
$true_slope = 0.3;
$store = array();
$promo = array();
$visits = array();
for ($s = 0; $s < $n_stores; $s++) {
    $n_days = 15 + (int)((mt_rand() / mt_getrandmax()) * 11);
    for ($d = 0; $d < $n_days; $d++) {
        $promo_intensity = (mt_rand() / mt_getrandmax()) * 3.0;
        $lam = exp($true_intercept + $store_effect[$s] + $true_slope * $promo_intensity);
        $l_thresh = exp(-$lam);
        $k = 0;
        $p = 1.0;
        do {
            $k += 1;
            $p *= (mt_rand() / mt_getrandmax());
        } while ($p > $l_thresh);
        $store[] = $s;
        $promo[] = $promo_intensity;
        $visits[] = $k - 1;
    }
}

$df = new DataFrame();
$df->add_numeric_column("store", dvector($store));
$df->add_numeric_column("promo", dvector($promo));
$df->add_numeric_column("visits", dvector($visits));

$store_model = new GLMM($df, "visits ~ promo + (1 | store)", "poisson");
$store_model->print_summary();

print("\nTrue generating values: intercept=$true_intercept, slope=$true_slope, random-intercept SD (log scale)=0.4\n");

print("\n--- BLUPs for a few stores ---\n");
$group_labels = $store_model->group_labels();
for ($idx = 0; $idx < 3; $idx++) {
    $re = $store_model->random_effects_for_group($idx);
    print("store " . $group_labels->get($idx) . ": intercept shift=" . $re->get(0) . "\n");
}

print("\n--- Prediction: population-level vs. store-adjusted ---\n");
$newdata_population = new DataFrame();
$newdata_population->add_numeric_column("promo", dvector(array(1.5)));
$newdata_store0 = new DataFrame();
$newdata_store0->add_numeric_column("promo", dvector(array(1.5)));
$newdata_store0->add_numeric_column("store", dvector(array(0.0)));
$pred_pop = $store_model->predict($newdata_population);
$pred_s0 = $store_model->predict($newdata_store0);
print("promo=1.5, unseen store:   " . $pred_pop->get(0) . " expected visits (fixed effects only)\n");
print("promo=1.5, store 0 (known): " . $pred_s0->get(0) . " expected visits (fixed effects + store 0's BLUP)\n");

?>
