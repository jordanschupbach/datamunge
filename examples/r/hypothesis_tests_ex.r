# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Hypothesis-test functions are free functions (not class methods), so they're already called
# in their natural flat form: t_test_two_sample(x, y), etc. Alternative is passed as a plain
# string matching the C++ enum member name ("TwoSided"/"Less"/"Greater") -- see
# datamunge_r_dollar_dispatch_bug.md for the general enum-handling pattern.
library(datamunger)

alt_names <- c("TwoSided" = "two.sided", "Less" = "less", "Greater" = "greater")

print_result <- function(label, r) {
  line <- paste0(label, ": statistic=", HypothesisTestResult_statistic_get(r))
  if (HypothesisTestResult_parameter1_get(r) != 0.0) {
    line <- paste0(line, ", df1=", HypothesisTestResult_parameter1_get(r))
  }
  if (HypothesisTestResult_parameter2_get(r) != 0.0) {
    line <- paste0(line, ", df2=", HypothesisTestResult_parameter2_get(r))
  }
  line <- paste0(line, ", p=", HypothesisTestResult_p_value_get(r),
                 " (", alt_names[[HypothesisTestResult_alternative_get(r)]], ")")
  if (HypothesisTestResult_has_conf_int_get(r)) {
    line <- paste0(line, ", CI=[", HypothesisTestResult_conf_int_lower_get(r), ", ",
                   HypothesisTestResult_conf_int_upper_get(r), "]")
  }
  line <- paste0(line, " -- ", HypothesisTestResult_method_get(r))
  cat(line, "\n")
}

column_for_species <- function(df, column, species) {
  n <- DataFrame_nrows(df)
  vals <- c()
  for (i in 0:(n - 1)) {
    if (DataFrame_string_at(df, "Species", i) == species) vals <- c(vals, DataFrame_numeric_at(df, column, i))
  }
  vals
}

iris <- DataFrame_iris()

setosa_petal <- column_for_species(iris, "Petal.Length", "setosa")
versicolor_petal <- column_for_species(iris, "Petal.Length", "versicolor")
virginica_petal <- column_for_species(iris, "Petal.Length", "virginica")

cat("=================== t-tests: petal length, setosa vs. versicolor ===================\n")
print_result("Welch two-sample t-test", t_test_two_sample(setosa_petal, versicolor_petal))
print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test(setosa_petal, versicolor_petal))

cat("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n")
all_petal <- c(setosa_petal, versicolor_petal, virginica_petal)
sizes <- c(length(setosa_petal), length(versicolor_petal), length(virginica_petal))
print_result("One-way ANOVA", one_way_anova(all_petal, sizes))
print_result("Kruskal-Wallis", kruskal_wallis_test(all_petal, sizes))

cat("\n=================== Correlation: sepal length vs. petal length ===================\n")
n <- DataFrame_nrows(iris)
sepal_length <- sapply(0:(n - 1), function(i) DataFrame_numeric_at(iris, "Sepal.Length", i))
petal_length <- sapply(0:(n - 1), function(i) DataFrame_numeric_at(iris, "Petal.Length", i))
print_result("Pearson correlation", pearson_correlation_test(sepal_length, petal_length))
print_result("Spearman correlation", spearman_correlation_test(sepal_length, petal_length))

cat("\n=================== F-test: petal length variance, setosa vs. virginica ===================\n")
print_result("F test", f_test_variance(setosa_petal, virginica_petal))

cat("\n=================== Normality: is sepal length normally distributed within setosa? ===================\n")
setosa_sepal <- column_for_species(iris, "Sepal.Length", "setosa")
print_result("Shapiro-Francia", shapiro_francia_test(setosa_sepal))
print_result("KS vs. fitted normal", ks_test_one_sample_normal(setosa_sepal, 5.006, 0.3525))

cat("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n")
median_all <- sort(all_petal)[floor(length(all_petal) / 2) + 1]
setosa_long <- sum(setosa_petal > median_all)
setosa_short <- length(setosa_petal) - setosa_long
versicolor_long <- sum(versicolor_petal > median_all)
versicolor_short <- length(versicolor_petal) - versicolor_long
cat("table: setosa=[", setosa_long, ",", setosa_short, "] versicolor=[", versicolor_long, ",", versicolor_short, "]\n", sep = "")
table <- c(setosa_long, setosa_short, versicolor_long, versicolor_short)
print_result("Chi-squared independence", chi_squared_test_independence(table, 2, 2))
print_result("Fisher's exact test", fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short))

cat("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n")
long_count <- sum(all_petal > median_all)
print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample(long_count, length(all_petal), 0.5))
print_result("Exact binomial test (vs 0.5)", binomial_test(long_count, length(all_petal), 0.5))
