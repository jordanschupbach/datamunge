using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static void PrintResult(string label, HypothesisTestResult r) {
    var line = $"{label}: statistic={r.statistic}";
    if (r.parameter1 != 0.0) line += $", df1={r.parameter1}";
    if (r.parameter2 != 0.0) line += $", df2={r.parameter2}";
    var altNames = new[] { "two.sided", "less", "greater" };
    line += $", p={r.p_value} ({altNames[(int)r.alternative]})";
    if (r.has_conf_int) line += $", CI=[{r.conf_int_lower}, {r.conf_int_upper}]";
    line += $" -- {r.method}";
    Console.WriteLine(line);
  }

  static List<double> ColumnForSpecies(DataFrame df, string column, string species) {
    var outp = new List<double>();
    uint n = df.nrows();
    for (uint i = 0; i < n; i++) {
      if (df.string_at("Species", i) == species) outp.Add(df.numeric_at(column, i));
    }
    return outp;
  }

  static void Main() {
    var iris = DataFrame.iris();

    var setosaPetal = ColumnForSpecies(iris, "Petal.Length", "setosa");
    var versicolorPetal = ColumnForSpecies(iris, "Petal.Length", "versicolor");
    var virginicaPetal = ColumnForSpecies(iris, "Petal.Length", "virginica");

    Console.WriteLine("=================== t-tests: petal length, setosa vs. versicolor ===================");
    PrintResult("Welch two-sample t-test", datamunge.t_test_two_sample(new DVector(setosaPetal), new DVector(versicolorPetal)));
    PrintResult("Wilcoxon rank-sum test", datamunge.wilcoxon_rank_sum_test(new DVector(setosaPetal), new DVector(versicolorPetal)));

    Console.WriteLine("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
    var allPetal = setosaPetal.Concat(versicolorPetal).Concat(virginicaPetal).ToList();
    var sizes = new SizeVector(new uint[] { (uint)setosaPetal.Count, (uint)versicolorPetal.Count, (uint)virginicaPetal.Count });
    PrintResult("One-way ANOVA", datamunge.one_way_anova(new DVector(allPetal), sizes));
    PrintResult("Kruskal-Wallis", datamunge.kruskal_wallis_test(new DVector(allPetal), sizes));

    Console.WriteLine("\n=================== Correlation: sepal length vs. petal length ===================");
    uint n = iris.nrows();
    var sepalLength = new List<double>();
    var petalLength = new List<double>();
    for (uint i = 0; i < n; i++) {
      sepalLength.Add(iris.numeric_at("Sepal.Length", i));
      petalLength.Add(iris.numeric_at("Petal.Length", i));
    }
    PrintResult("Pearson correlation", datamunge.pearson_correlation_test(new DVector(sepalLength), new DVector(petalLength)));
    PrintResult("Spearman correlation", datamunge.spearman_correlation_test(new DVector(sepalLength), new DVector(petalLength)));

    Console.WriteLine("\n=================== F-test: petal length variance, setosa vs. virginica ===================");
    PrintResult("F test", datamunge.f_test_variance(new DVector(setosaPetal), new DVector(virginicaPetal)));

    Console.WriteLine("\n=================== Normality: is sepal length normally distributed within setosa? ===================");
    var setosaSepal = ColumnForSpecies(iris, "Sepal.Length", "setosa");
    PrintResult("Shapiro-Francia", datamunge.shapiro_francia_test(new DVector(setosaSepal)));
    PrintResult("KS vs. fitted normal", datamunge.ks_test_one_sample_normal(new DVector(setosaSepal), 5.006, 0.3525));

    Console.WriteLine("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================");
    var sortedAll = allPetal.OrderBy(x => x).ToList();
    double medianAll = sortedAll[allPetal.Count / 2];
    int setosaLong = setosaPetal.Count(x => x > medianAll);
    int setosaShort = setosaPetal.Count - setosaLong;
    int versicolorLong = versicolorPetal.Count(x => x > medianAll);
    int versicolorShort = versicolorPetal.Count - versicolorLong;
    Console.WriteLine($"table: setosa=[{setosaLong},{setosaShort}] versicolor=[{versicolorLong},{versicolorShort}]");
    var tableVec = new DVector(new double[] { setosaLong, setosaShort, versicolorLong, versicolorShort });
    PrintResult("Chi-squared independence", datamunge.chi_squared_test_independence(tableVec, 2, 2));
    PrintResult("Fisher's exact test", datamunge.fisher_exact_test_2x2((uint)setosaLong, (uint)setosaShort, (uint)versicolorLong, (uint)versicolorShort));

    Console.WriteLine("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================");
    int longCount = allPetal.Count(x => x > medianAll);
    PrintResult("One-sample proportion test (vs 0.5)", datamunge.proportion_test_one_sample((uint)longCount, (uint)allPetal.Count, 0.5));
    PrintResult("Exact binomial test (vs 0.5)", datamunge.binomial_test((uint)longCount, (uint)allPetal.Count, 0.5));
  }
}
