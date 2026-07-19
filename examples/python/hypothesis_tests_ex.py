from pydatamunge import datamunge as dm

_ALT_NAMES = {dm.Alternative_Less: "less", dm.Alternative_Greater: "greater", dm.Alternative_TwoSided: "two.sided"}


def print_result(label, r):
    line = f"{label}: statistic={r.statistic}"
    if r.parameter1 != 0.0:
        line += f", df1={r.parameter1}"
    if r.parameter2 != 0.0:
        line += f", df2={r.parameter2}"
    line += f", p={r.p_value} ({_ALT_NAMES[r.alternative]})"
    if r.has_conf_int:
        line += f", CI=[{r.conf_int_lower}, {r.conf_int_upper}]"
    line += f" -- {r.method}"
    print(line)


def column_for_species(df, column, species):
    return [df.numeric_at(column, i) for i in range(df.nrows()) if df.string_at("Species", i) == species]


iris = dm.DataFrame.iris()

setosa_petal = column_for_species(iris, "Petal.Length", "setosa")
versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor")
virginica_petal = column_for_species(iris, "Petal.Length", "virginica")

print("=================== t-tests: petal length, setosa vs. versicolor ===================")
print_result("Welch two-sample t-test", dm.t_test_two_sample(setosa_petal, versicolor_petal))
print_result("Wilcoxon rank-sum test", dm.wilcoxon_rank_sum_test(setosa_petal, versicolor_petal))

print("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
all_petal = setosa_petal + versicolor_petal + virginica_petal
sizes = [len(setosa_petal), len(versicolor_petal), len(virginica_petal)]
print_result("One-way ANOVA", dm.one_way_anova(all_petal, sizes))
print_result("Kruskal-Wallis", dm.kruskal_wallis_test(all_petal, sizes))

print("\n=================== Correlation: sepal length vs. petal length ===================")
sepal_length = [iris.numeric_at("Sepal.Length", i) for i in range(iris.nrows())]
petal_length = [iris.numeric_at("Petal.Length", i) for i in range(iris.nrows())]
print_result("Pearson correlation", dm.pearson_correlation_test(sepal_length, petal_length))
print_result("Spearman correlation", dm.spearman_correlation_test(sepal_length, petal_length))

print("\n=================== F-test: petal length variance, setosa vs. virginica ===================")
print_result("F test", dm.f_test_variance(setosa_petal, virginica_petal))

print("\n=================== Normality: is sepal length normally distributed within setosa? "
      "===================")
setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa")
print_result("Shapiro-Francia", dm.shapiro_francia_test(setosa_sepal))
print_result("KS vs. fitted normal", dm.ks_test_one_sample_normal(setosa_sepal, 5.006, 0.3525))

print("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? "
      "===================")
median_all = sorted(all_petal)[len(all_petal) // 2]
setosa_long = sum(1 for v in setosa_petal if v > median_all)
setosa_short = len(setosa_petal) - setosa_long
versicolor_long = sum(1 for v in versicolor_petal if v > median_all)
versicolor_short = len(versicolor_petal) - versicolor_long
print(f"table: setosa=[{setosa_long},{setosa_short}] versicolor=[{versicolor_long},{versicolor_short}]")
table = [float(setosa_long), float(setosa_short), float(versicolor_long), float(versicolor_short)]
print_result("Chi-squared independence", dm.chi_squared_test_independence(table, 2, 2))
print_result("Fisher's exact test", dm.fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short))

print("\n=================== Proportion / binomial: fraction of \"long\" petals overall "
      "===================")
long_count = sum(1 for v in all_petal if v > median_all)
print_result("One-sample proportion test (vs 0.5)", dm.proportion_test_one_sample(long_count, len(all_petal), 0.5))
print_result("Exact binomial test (vs 0.5)", dm.binomial_test(long_count, len(all_petal), 0.5))
