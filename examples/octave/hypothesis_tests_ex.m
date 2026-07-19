1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function v = szv(t)
  datamunge;
  v = SizeVector();
  for i = 1:numel(t)
    SizeVector_push_back(v, t(i));
  end
endfunction

function print_result(label, r)
  datamunge;
  line = sprintf("%s: statistic=%g", label, HypothesisTestResult_statistic_get(r));
  if HypothesisTestResult_parameter1_get(r) != 0.0
    line = [line, sprintf(", df1=%g", HypothesisTestResult_parameter1_get(r))];
  end
  if HypothesisTestResult_parameter2_get(r) != 0.0
    line = [line, sprintf(", df2=%g", HypothesisTestResult_parameter2_get(r))];
  end
  alt_names = {"two.sided", "less", "greater"};
  line = [line, sprintf(", p=%g (%s)", HypothesisTestResult_p_value_get(r), alt_names{HypothesisTestResult_alternative_get(r) + 1})];
  if HypothesisTestResult_has_conf_int_get(r)
    line = [line, sprintf(", CI=[%g, %g]", HypothesisTestResult_conf_int_lower_get(r), HypothesisTestResult_conf_int_upper_get(r))];
  end
  line = [line, sprintf(" -- %s", HypothesisTestResult_method_get(r))];
  printf("%s\n", line);
endfunction

function out = column_for_species(df, column, species)
  datamunge;
  out = [];
  n = DataFrame_nrows(df);
  for i = 0:(n - 1)
    if strcmp(DataFrame_string_at(df, "Species", i), species)
      out(end + 1) = DataFrame_numeric_at(df, column, i);
    end
  end
endfunction

iris = DataFrame_iris();

setosa_petal = column_for_species(iris, "Petal.Length", "setosa");
versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor");
virginica_petal = column_for_species(iris, "Petal.Length", "virginica");

printf("=================== t-tests: petal length, setosa vs. versicolor ===================\n");
print_result("Welch two-sample t-test", t_test_two_sample(dv(setosa_petal), dv(versicolor_petal)));
print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test(dv(setosa_petal), dv(versicolor_petal)));

printf("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n");
all_petal = [setosa_petal, versicolor_petal, virginica_petal];
sizes = szv([numel(setosa_petal), numel(versicolor_petal), numel(virginica_petal)]);
print_result("One-way ANOVA", one_way_anova(dv(all_petal), sizes));
print_result("Kruskal-Wallis", kruskal_wallis_test(dv(all_petal), sizes));

printf("\n=================== Correlation: sepal length vs. petal length ===================\n");
n = DataFrame_nrows(iris);
sepal_length = [];
petal_length = [];
for i = 0:(n - 1)
  sepal_length(end + 1) = DataFrame_numeric_at(iris, "Sepal.Length", i);
  petal_length(end + 1) = DataFrame_numeric_at(iris, "Petal.Length", i);
end
print_result("Pearson correlation", pearson_correlation_test(dv(sepal_length), dv(petal_length)));
print_result("Spearman correlation", spearman_correlation_test(dv(sepal_length), dv(petal_length)));

printf("\n=================== F-test: petal length variance, setosa vs. virginica ===================\n");
print_result("F test", f_test_variance(dv(setosa_petal), dv(virginica_petal)));

printf("\n=================== Normality: is sepal length normally distributed within setosa? ===================\n");
setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa");
print_result("Shapiro-Francia", shapiro_francia_test(dv(setosa_sepal)));
print_result("KS vs. fitted normal", ks_test_one_sample_normal(dv(setosa_sepal), 5.006, 0.3525));

printf("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n");
sorted_all = sort(all_petal);
median_all = sorted_all(floor(numel(all_petal) / 2) + 1);
setosa_long = sum(setosa_petal > median_all);
setosa_short = numel(setosa_petal) - setosa_long;
versicolor_long = sum(versicolor_petal > median_all);
versicolor_short = numel(versicolor_petal) - versicolor_long;
printf("table: setosa=[%d,%d] versicolor=[%d,%d]\n", setosa_long, setosa_short, versicolor_long, versicolor_short);
table_vec = dv([setosa_long, setosa_short, versicolor_long, versicolor_short]);
print_result("Chi-squared independence", chi_squared_test_independence(table_vec, 2, 2));
print_result("Fisher's exact test", fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short));

printf("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n");
long_count = sum(all_petal > median_all);
print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample(long_count, numel(all_petal), 0.5));
print_result("Exact binomial test (vs 0.5)", binomial_test(long_count, numel(all_petal), 0.5));
