# CMD: just TARGET=hypothesis_tests_ex run-ruby
# Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
require "datamunge"

DM = Datamunge

def print_result(label, r)
  alternative = { Less: "less", Greater: "greater" }.fetch(r.alternative, "two.sided")
  line = +"#{label}: statistic=#{r.statistic}"
  line << ", df1=#{r.parameter1}" unless r.parameter1.zero?
  line << ", df2=#{r.parameter2}" unless r.parameter2.zero?
  line << ", p=#{r.p_value} (#{alternative})"
  line << ", CI=[#{r.conf_int_lower}, #{r.conf_int_upper}]" if r.has_conf_int
  puts "#{line} -- #{r.method_}"
end

def column(df, column, species = nil)
  (0...df.nrows).filter_map do |i|
    df.numeric_at(column, i) if species.nil? || df.string_at("Species", i) == species
  end
end

iris = DM::Data.iris

setosa_petal = column(iris, "Petal.Length", "setosa")
versicolor_petal = column(iris, "Petal.Length", "versicolor")
virginica_petal = column(iris, "Petal.Length", "virginica")

puts "=================== t-tests: petal length, setosa vs. versicolor ==================="
print_result("Welch two-sample t-test", DM.t_test_two_sample(setosa_petal, versicolor_petal))
print_result("Wilcoxon rank-sum test", DM.wilcoxon_rank_sum_test(setosa_petal, versicolor_petal))

puts "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================="
all_petal = setosa_petal + versicolor_petal + virginica_petal
sizes = [setosa_petal, versicolor_petal, virginica_petal].map(&:size)
print_result("One-way ANOVA", DM.one_way_anova(all_petal, sizes))
print_result("Kruskal-Wallis", DM.kruskal_wallis_test(all_petal, sizes))

puts "\n=================== Correlation: sepal length vs. petal length ==================="
sepal_length = column(iris, "Sepal.Length")
petal_length = column(iris, "Petal.Length")
print_result("Pearson correlation", DM.pearson_correlation_test(sepal_length, petal_length))
print_result("Spearman correlation", DM.spearman_correlation_test(sepal_length, petal_length))

puts "\n=================== F-test: petal length variance, setosa vs. virginica ==================="
print_result("F test", DM.f_test_variance(setosa_petal, virginica_petal))

puts "\n=================== Normality: is sepal length normally distributed within setosa? ==================="
setosa_sepal = column(iris, "Sepal.Length", "setosa")
print_result("Shapiro-Francia", DM.shapiro_francia_test(setosa_sepal))
print_result("KS vs. fitted normal", DM.ks_test_one_sample_normal(setosa_sepal, 5.006, 0.3525))

puts "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ==================="
median_all = all_petal.sort[all_petal.size / 2]
count_long = ->(v) { v.count { |x| x > median_all } }
setosa_long = count_long.(setosa_petal)
setosa_short = setosa_petal.size - setosa_long
versicolor_long = count_long.(versicolor_petal)
versicolor_short = versicolor_petal.size - versicolor_long
puts "table: setosa=[#{setosa_long},#{setosa_short}] versicolor=[#{versicolor_long},#{versicolor_short}]"
table = [setosa_long, setosa_short, versicolor_long, versicolor_short].map(&:to_f)
print_result("Chi-squared independence", DM.chi_squared_test_independence(table, 2, 2))
print_result("Fisher's exact test", DM.fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short))

puts "\n=================== Proportion / binomial: fraction of \"long\" petals overall ==================="
long_count = count_long.(all_petal)
n = all_petal.size
print_result("One-sample proportion test (vs 0.5)", DM.proportion_test_one_sample(long_count, n, 0.5))
print_result("Exact binomial test (vs 0.5)", DM.binomial_test(long_count, n, 0.5))
