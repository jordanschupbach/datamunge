require "octruby"

ALT_NAMES = {
  Datamunge::Alternative_Less => "less",
  Datamunge::Alternative_Greater => "greater",
  Datamunge::Alternative_TwoSided => "two.sided",
}.freeze

def print_result(label, r)
  line = "#{label}: statistic=#{r.statistic}"
  line += ", df1=#{r.parameter1}" if r.parameter1 != 0.0
  line += ", df2=#{r.parameter2}" if r.parameter2 != 0.0
  line += ", p=#{r.p_value} (#{ALT_NAMES[r.alternative]})"
  line += ", CI=[#{r.conf_int_lower}, #{r.conf_int_upper}]" if r.has_conf_int
  line += " -- #{r.method}"
  puts line
end

def column_for_species(df, column, species)
  (0...df.nrows).select { |i| df.string_at("Species", i) == species }.map { |i| df.numeric_at(column, i) }
end

iris = Datamunge::DataFrame.iris

setosa_petal = column_for_species(iris, "Petal.Length", "setosa")
versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor")
virginica_petal = column_for_species(iris, "Petal.Length", "virginica")

puts "=================== t-tests: petal length, setosa vs. versicolor ==================="
print_result("Welch two-sample t-test", Datamunge.t_test_two_sample(setosa_petal, versicolor_petal))
print_result("Wilcoxon rank-sum test", Datamunge.wilcoxon_rank_sum_test(setosa_petal, versicolor_petal))

puts "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================="
all_petal = setosa_petal + versicolor_petal + virginica_petal
sizes = [setosa_petal.length, versicolor_petal.length, virginica_petal.length]
print_result("One-way ANOVA", Datamunge.one_way_anova(all_petal, sizes))
print_result("Kruskal-Wallis", Datamunge.kruskal_wallis_test(all_petal, sizes))

puts "\n=================== Correlation: sepal length vs. petal length ==================="
sepal_length = (0...iris.nrows).map { |i| iris.numeric_at("Sepal.Length", i) }
petal_length = (0...iris.nrows).map { |i| iris.numeric_at("Petal.Length", i) }
print_result("Pearson correlation", Datamunge.pearson_correlation_test(sepal_length, petal_length))
print_result("Spearman correlation", Datamunge.spearman_correlation_test(sepal_length, petal_length))

puts "\n=================== F-test: petal length variance, setosa vs. virginica ==================="
print_result("F test", Datamunge.f_test_variance(setosa_petal, virginica_petal))

puts "\n=================== Normality: is sepal length normally distributed within setosa? ===================\n"
setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa")
print_result("Shapiro-Francia", Datamunge.shapiro_francia_test(setosa_sepal))
print_result("KS vs. fitted normal", Datamunge.ks_test_one_sample_normal(setosa_sepal, 5.006, 0.3525))

puts "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================\n"
median_all = all_petal.sort[all_petal.length / 2]
setosa_long = setosa_petal.count { |v| v > median_all }
setosa_short = setosa_petal.length - setosa_long
versicolor_long = versicolor_petal.count { |v| v > median_all }
versicolor_short = versicolor_petal.length - versicolor_long
puts "table: setosa=[#{setosa_long},#{setosa_short}] versicolor=[#{versicolor_long},#{versicolor_short}]"
table = [setosa_long.to_f, setosa_short.to_f, versicolor_long.to_f, versicolor_short.to_f]
print_result("Chi-squared independence", Datamunge.chi_squared_test_independence(table, 2, 2))
print_result("Fisher's exact test", Datamunge.fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short))

puts "\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================\n"
long_count = all_petal.count { |v| v > median_all }
print_result("One-sample proportion test (vs 0.5)", Datamunge.proportion_test_one_sample(long_count, all_petal.length, 0.5))
print_result("Exact binomial test (vs 0.5)", Datamunge.binomial_test(long_count, all_petal.length, 0.5))
