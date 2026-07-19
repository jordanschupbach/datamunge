module app;

import std.stdio : writeln;
import std.algorithm : sort, count;
import std.conv : to;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

SizeVector szv(size_t[] t) {
  auto v = new SizeVector();
  foreach (x; t) v.push_back(x);
  return v;
}

string altName(Alternative a) {
  final switch (a) {
    case Alternative.TwoSided: return "two.sided";
    case Alternative.Less: return "less";
    case Alternative.Greater: return "greater";
  }
}

void print_result(string label, HypothesisTestResult r) {
  string line = label ~ ": statistic=" ~ to!string(r.statistic());
  if (r.parameter1() != 0.0) line ~= ", df1=" ~ to!string(r.parameter1());
  if (r.parameter2() != 0.0) line ~= ", df2=" ~ to!string(r.parameter2());
  line ~= ", p=" ~ to!string(r.p_value()) ~ " (" ~ altName(r.alternative()) ~ ")";
  if (r.has_conf_int()) line ~= ", CI=[" ~ to!string(r.conf_int_lower()) ~ ", " ~ to!string(r.conf_int_upper()) ~ "]";
  line ~= " -- " ~ r.method();
  writeln(line);
}

double[] column_for_species(DataFrame df, string column, string species) {
  double[] out_;
  auto n = df.nrows();
  for (size_t i = 0; i < n; i++) {
    if (df.string_at("Species", i) == species) out_ ~= df.numeric_at(column, i);
  }
  return out_;
}

void main() {
  auto iris = DataFrame.iris();

  auto setosa_petal = column_for_species(iris, "Petal.Length", "setosa");
  auto versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor");
  auto virginica_petal = column_for_species(iris, "Petal.Length", "virginica");

  writeln("=================== t-tests: petal length, setosa vs. versicolor ===================");
  print_result("Welch two-sample t-test", t_test_two_sample(dv(setosa_petal), dv(versicolor_petal)));
  print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test(dv(setosa_petal), dv(versicolor_petal)));

  writeln("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
  double[] all_petal = setosa_petal ~ versicolor_petal ~ virginica_petal;
  auto sizes = szv([setosa_petal.length, versicolor_petal.length, virginica_petal.length]);
  print_result("One-way ANOVA", one_way_anova(dv(all_petal), sizes));
  print_result("Kruskal-Wallis", kruskal_wallis_test(dv(all_petal), sizes));

  writeln("\n=================== Correlation: sepal length vs. petal length ===================");
  double[] sepal_length;
  double[] petal_length;
  auto n = iris.nrows();
  for (size_t i = 0; i < n; i++) {
    sepal_length ~= iris.numeric_at("Sepal.Length", i);
    petal_length ~= iris.numeric_at("Petal.Length", i);
  }
  print_result("Pearson correlation", pearson_correlation_test(dv(sepal_length), dv(petal_length)));
  print_result("Spearman correlation", spearman_correlation_test(dv(sepal_length), dv(petal_length)));

  writeln("\n=================== F-test: petal length variance, setosa vs. virginica ===================");
  print_result("F test", f_test_variance(dv(setosa_petal), dv(virginica_petal)));

  writeln("\n=================== Normality: is sepal length normally distributed within setosa? ===================");
  auto setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa");
  print_result("Shapiro-Francia", shapiro_francia_test(dv(setosa_sepal)));
  print_result("KS vs. fitted normal", ks_test_one_sample_normal(dv(setosa_sepal), 5.006, 0.3525));

  writeln("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================");
  auto sorted_all = all_petal.dup;
  sort(sorted_all);
  double median_all = sorted_all[all_petal.length / 2];
  int setosa_long = cast(int) count!(x => x > median_all)(setosa_petal);
  int setosa_short = cast(int)(setosa_petal.length - setosa_long);
  int versicolor_long = cast(int) count!(x => x > median_all)(versicolor_petal);
  int versicolor_short = cast(int)(versicolor_petal.length - versicolor_long);
  writeln("table: setosa=[", setosa_long, ",", setosa_short, "] versicolor=[", versicolor_long, ",", versicolor_short, "]");
  auto table_vec = dv([cast(double) setosa_long, cast(double) setosa_short, cast(double) versicolor_long, cast(double) versicolor_short]);
  print_result("Chi-squared independence", chi_squared_test_independence(table_vec, 2, 2));
  print_result("Fisher's exact test", fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short));

  writeln("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================");
  int long_count = cast(int) count!(x => x > median_all)(all_petal);
  print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample(long_count, cast(int) all_petal.length, 0.5));
  print_result("Exact binomial test (vs 0.5)", binomial_test(long_count, cast(int) all_petal.length, 0.5));
}
