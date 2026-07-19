package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.SizeVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.HypothesisTestResult;
import js.datamunge.jdatamunge.datamunge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class HypothesisTestsEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static void printResult(String label, HypothesisTestResult r) {
    StringBuilder line = new StringBuilder(label + ": statistic=" + r.getStatistic());
    if (r.getParameter1() != 0.0) line.append(", df1=").append(r.getParameter1());
    if (r.getParameter2() != 0.0) line.append(", df2=").append(r.getParameter2());
    String[] altNames = {"two.sided", "less", "greater"};
    line.append(", p=").append(r.getP_value()).append(" (").append(altNames[r.getAlternative().swigValue()]).append(")");
    if (r.getHas_conf_int()) line.append(", CI=[").append(r.getConf_int_lower()).append(", ").append(r.getConf_int_upper()).append("]");
    line.append(" -- ").append(r.getMethod());
    System.out.println(line);
  }

  static List<Double> columnForSpecies(DataFrame df, String column, String species) {
    List<Double> out = new ArrayList<>();
    long n = df.nrows();
    for (long i = 0; i < n; i++) {
      if (df.string_at("Species", i).equals(species)) out.add(df.numeric_at(column, i));
    }
    return out;
  }

  public static void run() {
    var iris = DataFrame.iris();

    var setosaPetal = columnForSpecies(iris, "Petal.Length", "setosa");
    var versicolorPetal = columnForSpecies(iris, "Petal.Length", "versicolor");
    var virginicaPetal = columnForSpecies(iris, "Petal.Length", "virginica");

    System.out.println("=================== t-tests: petal length, setosa vs. versicolor ===================");
    printResult("Welch two-sample t-test", datamunge.t_test_two_sample(new DVector(setosaPetal), new DVector(versicolorPetal)));
    printResult("Wilcoxon rank-sum test", datamunge.wilcoxon_rank_sum_test(new DVector(setosaPetal), new DVector(versicolorPetal)));

    System.out.println("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
    List<Double> allPetal = new ArrayList<>();
    allPetal.addAll(setosaPetal);
    allPetal.addAll(versicolorPetal);
    allPetal.addAll(virginicaPetal);
    var sizes = new SizeVector(new long[] {setosaPetal.size(), versicolorPetal.size(), virginicaPetal.size()});
    printResult("One-way ANOVA", datamunge.one_way_anova(new DVector(allPetal), sizes));
    printResult("Kruskal-Wallis", datamunge.kruskal_wallis_test(new DVector(allPetal), sizes));

    System.out.println("\n=================== Correlation: sepal length vs. petal length ===================");
    long n = iris.nrows();
    List<Double> sepalLength = new ArrayList<>();
    List<Double> petalLength = new ArrayList<>();
    for (long i = 0; i < n; i++) {
      sepalLength.add(iris.numeric_at("Sepal.Length", i));
      petalLength.add(iris.numeric_at("Petal.Length", i));
    }
    printResult("Pearson correlation", datamunge.pearson_correlation_test(new DVector(sepalLength), new DVector(petalLength)));
    printResult("Spearman correlation", datamunge.spearman_correlation_test(new DVector(sepalLength), new DVector(petalLength)));

    System.out.println("\n=================== F-test: petal length variance, setosa vs. virginica ===================");
    printResult("F test", datamunge.f_test_variance(new DVector(setosaPetal), new DVector(virginicaPetal)));

    System.out.println("\n=================== Normality: is sepal length normally distributed within setosa? ===================");
    var setosaSepal = columnForSpecies(iris, "Sepal.Length", "setosa");
    printResult("Shapiro-Francia", datamunge.shapiro_francia_test(new DVector(setosaSepal)));
    printResult("KS vs. fitted normal", datamunge.ks_test_one_sample_normal(new DVector(setosaSepal), 5.006, 0.3525));

    System.out.println("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================");
    List<Double> sortedAll = new ArrayList<>(allPetal);
    Collections.sort(sortedAll);
    double medianAll = sortedAll.get(allPetal.size() / 2);
    long setosaLong = setosaPetal.stream().filter(x -> x > medianAll).count();
    long setosaShort = setosaPetal.size() - setosaLong;
    long versicolorLong = versicolorPetal.stream().filter(x -> x > medianAll).count();
    long versicolorShort = versicolorPetal.size() - versicolorLong;
    System.out.println("table: setosa=[" + setosaLong + "," + setosaShort + "] versicolor=[" + versicolorLong + "," + versicolorShort + "]");
    var tableVec = new DVector(new double[] {setosaLong, setosaShort, versicolorLong, versicolorShort});
    printResult("Chi-squared independence", datamunge.chi_squared_test_independence(tableVec, 2, 2));
    printResult("Fisher's exact test", datamunge.fisher_exact_test_2x2(setosaLong, setosaShort, versicolorLong, versicolorShort));

    System.out.println("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================");
    long longCount = allPetal.stream().filter(x -> x > medianAll).count();
    printResult("One-sample proportion test (vs 0.5)", datamunge.proportion_test_one_sample(longCount, allPetal.size(), 0.5));
    printResult("Exact binomial test (vs 0.5)", datamunge.binomial_test(longCount, allPetal.size(), 0.5));
  }
}
