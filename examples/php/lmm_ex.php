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

print("=================== Random intercept on a real dataset (penguins) ===================\n");
$penguins = DataFrame::penguins();
$species_model = new LMM($penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
$species_model->print_summary();

print("\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n");
$n_schools = 30;
$school_intercept = array();
$school_slope = array();
for ($i = 0; $i < $n_schools; $i++) {
    $school_intercept[] = randn() * 6.0;
    $school_slope[] = randn() * 1.2;
}

$true_intercept = 60.0;
$true_slope = 3.0;
$school = array();
$study_hours = array();
$score = array();
for ($s = 0; $s < $n_schools; $s++) {
    $n_students = 15 + (int)((mt_rand() / mt_getrandmax()) * 21);
    for ($j = 0; $j < $n_students; $j++) {
        $hours = (mt_rand() / mt_getrandmax()) * 10.0;
        $noise = randn() * 4.0;
        $s_val = $true_intercept + $school_intercept[$s] + ($true_slope + $school_slope[$s]) * $hours + $noise;
        $school[] = $s;
        $study_hours[] = $hours;
        $score[] = $s_val;
    }
}

$df = new DataFrame();
$df->add_numeric_column("school", dvector($school));
$df->add_numeric_column("study_hours", dvector($study_hours));
$df->add_numeric_column("score", dvector($score));

$model = new LMM($df, "score ~ study_hours + (1 + study_hours | school)");
$model->print_summary();

print("\nTrue generating values: intercept=$true_intercept, slope=$true_slope, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n");

print("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n");
$group_labels = $model->group_labels();
for ($idx = 0; $idx < 3; $idx++) {
    $re = $model->random_effects_for_group($idx);
    print("school " . $group_labels->get($idx) . ": intercept shift=" . $re->get(0) . ", slope shift=" . $re->get(1) . "\n");
}

print("\n--- Prediction: population-level vs. school-adjusted ---\n");
$newdata_population = new DataFrame();
$newdata_population->add_numeric_column("study_hours", dvector(array(5.0)));
$newdata_school0 = new DataFrame();
$newdata_school0->add_numeric_column("study_hours", dvector(array(5.0)));
$newdata_school0->add_numeric_column("school", dvector(array(0.0)));
$pred_pop = $model->predict($newdata_population);
$pred_s0 = $model->predict($newdata_school0);
print("5 study hours, unseen school:      " . $pred_pop->get(0) . " (fixed effects only)\n");
print("5 study hours, school 0 (known):    " . $pred_s0->get(0) . " (fixed effects + school 0's BLUP)\n");

?>
