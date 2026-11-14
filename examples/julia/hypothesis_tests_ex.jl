# CMD: just TARGET=hypothesis_tests_ex run-julia

# Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
using Datamunge

const alt_names = Dict(:Less => "less", :Greater => "greater", :TwoSided => "two.sided")

function print_result(label, r)
    print("$label: statistic=$(r.statistic)")
    r.parameter1 != 0 && print(", df1=$(r.parameter1)")
    r.parameter2 != 0 && print(", df2=$(r.parameter2)")
    print(", p=$(r.p_value) ($(alt_names[r.alternative]))")
    r.has_conf_int && print(", CI=[$(r.conf_int_lower), $(r.conf_int_upper)]")
    println(" -- $(r.method)")
end

column_for_species(df, column, species) =
    [numeric_at(df, column, i) for i in 0:nrows(df)-1 if string_at(df, "Species", i) == species]

const iris = Datamunge.Data.iris()

const setosa_petal = column_for_species(iris, "Petal.Length", "setosa")
const versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor")
const virginica_petal = column_for_species(iris, "Petal.Length", "virginica")

println("=================== t-tests: petal length, setosa vs. versicolor ===================")
print_result("Welch two-sample t-test", t_test_two_sample(setosa_petal, versicolor_petal))
print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test(setosa_petal, versicolor_petal))

println("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
const all_petal = vcat(setosa_petal, versicolor_petal, virginica_petal)
let sizes = length.([setosa_petal, versicolor_petal, virginica_petal])
    print_result("One-way ANOVA", one_way_anova(all_petal, sizes))
    print_result("Kruskal-Wallis", kruskal_wallis_test(all_petal, sizes))
end

println("\n=================== Correlation: sepal length vs. petal length ===================")
let sepal_length = [numeric_at(iris, "Sepal.Length", i) for i in 0:nrows(iris)-1],
    petal_length = [numeric_at(iris, "Petal.Length", i) for i in 0:nrows(iris)-1]
    print_result("Pearson correlation", pearson_correlation_test(sepal_length, petal_length))
    print_result("Spearman correlation", spearman_correlation_test(sepal_length, petal_length))
end

println("\n=================== F-test: petal length variance, setosa vs. virginica ===================")
print_result("F test", f_test_variance(setosa_petal, virginica_petal))

println("\n=================== Normality: is sepal length normally distributed within setosa? ===================")
let setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa")
    print_result("Shapiro-Francia", shapiro_francia_test(setosa_sepal))
    print_result("KS vs. fitted normal", ks_test_one_sample_normal(setosa_sepal, 5.006, 0.3525))
end

println("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================")
const median_all = sort(all_petal)[length(all_petal) ÷ 2 + 1]
count_long(v) = count(>(median_all), v)
let setosa_long = count_long(setosa_petal), setosa_short = length(setosa_petal) - setosa_long,
    versicolor_long = count_long(versicolor_petal), versicolor_short = length(versicolor_petal) - versicolor_long
    println("table: setosa=[$setosa_long,$setosa_short] versicolor=[$versicolor_long,$versicolor_short]")
    table = Float64[setosa_long, setosa_short, versicolor_long, versicolor_short]
    print_result("Chi-squared independence", chi_squared_test_independence(table, 2, 2))
    print_result("Fisher's exact test", fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short))
end

println("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================")
let long_count = count_long(all_petal), n = length(all_petal)
    print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample(long_count, n, 0.5))
    print_result("Exact binomial test (vs 0.5)", binomial_test(long_count, n, 0.5))
end
