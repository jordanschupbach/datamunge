<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function szvector($values) {
    $out = new SizeVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function print_result($label, $r) {
    $line = "$label: statistic=" . $r->statistic;
    if ($r->parameter1 != 0.0) $line .= ", df1=" . $r->parameter1;
    if ($r->parameter2 != 0.0) $line .= ", df2=" . $r->parameter2;
    $alt_names = array("two.sided", "less", "greater");
    $line .= ", p=" . $r->p_value . " (" . $alt_names[$r->alternative] . ")";
    if ($r->has_conf_int) $line .= ", CI=[" . $r->conf_int_lower . ", " . $r->conf_int_upper . "]";
    $line .= " -- " . $r->method;
    print($line . "\n");
}

function column_for_species($df, $column, $species) {
    $out = array();
    $n = $df->nrows();
    for ($i = 0; $i < $n; $i++) {
        if ($df->string_at("Species", $i) === $species) $out[] = $df->numeric_at($column, $i);
    }
    return $out;
}

$iris = DataFrame::iris();

$setosa_petal = column_for_species($iris, "Petal.Length", "setosa");
$versicolor_petal = column_for_species($iris, "Petal.Length", "versicolor");
$virginica_petal = column_for_species($iris, "Petal.Length", "virginica");

print("=================== t-tests: petal length, setosa vs. versicolor ===================\n");
print_result("Welch two-sample t-test", t_test_two_sample(dvector($setosa_petal), dvector($versicolor_petal)));
print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test(dvector($setosa_petal), dvector($versicolor_petal)));

print("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n");
$all_petal = array_merge($setosa_petal, $versicolor_petal, $virginica_petal);
$sizes = szvector(array(count($setosa_petal), count($versicolor_petal), count($virginica_petal)));
print_result("One-way ANOVA", one_way_anova(dvector($all_petal), $sizes));
print_result("Kruskal-Wallis", kruskal_wallis_test(dvector($all_petal), $sizes));

print("\n=================== Correlation: sepal length vs. petal length ===================\n");
$n = $iris->nrows();
$sepal_length = array();
$petal_length = array();
for ($i = 0; $i < $n; $i++) {
    $sepal_length[] = $iris->numeric_at("Sepal.Length", $i);
    $petal_length[] = $iris->numeric_at("Petal.Length", $i);
}
print_result("Pearson correlation", pearson_correlation_test(dvector($sepal_length), dvector($petal_length)));
print_result("Spearman correlation", spearman_correlation_test(dvector($sepal_length), dvector($petal_length)));

print("\n=================== F-test: petal length variance, setosa vs. virginica ===================\n");
print_result("F test", f_test_variance(dvector($setosa_petal), dvector($virginica_petal)));

print("\n=================== Normality: is sepal length normally distributed within setosa? ===================\n");
$setosa_sepal = column_for_species($iris, "Sepal.Length", "setosa");
print_result("Shapiro-Francia", shapiro_francia_test(dvector($setosa_sepal)));
print_result("KS vs. fitted normal", ks_test_one_sample_normal(dvector($setosa_sepal), 5.006, 0.3525));

print("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n");
$sorted_all = $all_petal;
sort($sorted_all);
$median_all = $sorted_all[intdiv(count($all_petal), 2)];
$setosa_long = count(array_filter($setosa_petal, function($x) use ($median_all) { return $x > $median_all; }));
$setosa_short = count($setosa_petal) - $setosa_long;
$versicolor_long = count(array_filter($versicolor_petal, function($x) use ($median_all) { return $x > $median_all; }));
$versicolor_short = count($versicolor_petal) - $versicolor_long;
print("table: setosa=[$setosa_long,$setosa_short] versicolor=[$versicolor_long,$versicolor_short]\n");
$table_vec = dvector(array($setosa_long, $setosa_short, $versicolor_long, $versicolor_short));
print_result("Chi-squared independence", chi_squared_test_independence($table_vec, 2, 2));
print_result("Fisher's exact test", fisher_exact_test_2x2($setosa_long, $setosa_short, $versicolor_long, $versicolor_short));

print("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n");
$long_count = count(array_filter($all_petal, function($x) use ($median_all) { return $x > $median_all; }));
print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample($long_count, count($all_petal), 0.5));
print_result("Exact binomial test (vs 0.5)", binomial_test($long_count, count($all_petal), 0.5));

?>
