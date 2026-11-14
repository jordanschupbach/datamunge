// CMD: just TARGET=hypothesis_tests_ex run-java
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
import datamunge.DataFrame;
import datamunge.Datamunge;
import datamunge.HypothesisTestResult;
import datamunge.data.Data;
import java.util.Arrays;
import java.util.stream.DoubleStream;

public class HypothesisTestsEx {
  static void printResult(String label, HypothesisTestResult r) {
    String alternative =
        switch (r.getAlternative()) {
          case TwoSided -> "two.sided";
          case Less -> "less";
          case Greater -> "greater";
        };
    var line = new StringBuilder(label + ": statistic=" + r.getStatistic());
    if (r.getParameter1() != 0) {
      line.append(", df1=").append(r.getParameter1());
    }
    if (r.getParameter2() != 0) {
      line.append(", df2=").append(r.getParameter2());
    }
    line.append(", p=").append(r.getPValue()).append(" (").append(alternative).append(")");
    if (r.getHasConfInt()) {
      line.append(", CI=[")
          .append(r.getConfIntLower())
          .append(", ")
          .append(r.getConfIntUpper())
          .append("]");
    }
    System.out.println(line + " -- " + r.getMethod());
    r.close();
  }

  // Values of a numeric column, optionally only for rows of one species (null = all rows).
  static double[] column(DataFrame df, String name, String species) {
    var values = DoubleStream.builder();
    for (long i = 0; i < df.nrows(); i++) {
      if (species == null || df.stringAt("Species", i).equals(species)) {
        values.add(df.numericAt(name, i));
      }
    }
    return values.build().toArray();
  }

  // Number of values strictly greater than threshold.
  static long countAbove(double[] values, double threshold) {
    return Arrays.stream(values).filter(x -> x > threshold).count();
  }

  public static void main(String[] args) {
    var iris = Data.iris();

    double[] setosaPetal = column(iris, "Petal.Length", "setosa");
    double[] versicolorPetal = column(iris, "Petal.Length", "versicolor");
    double[] virginicaPetal = column(iris, "Petal.Length", "virginica");

    System.out.println(
        "=================== t-tests: petal length, setosa vs. versicolor ===================");
    printResult("Welch two-sample t-test", Datamunge.tTestTwoSample(setosaPetal, versicolorPetal));
    printResult(
        "Wilcoxon rank-sum test", Datamunge.wilcoxonRankSumTest(setosaPetal, versicolorPetal));

    System.out.println(
        "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species"
            + " ===================");
    double[] allPetal =
        DoubleStream.concat(
                DoubleStream.concat(Arrays.stream(setosaPetal), Arrays.stream(versicolorPetal)),
                Arrays.stream(virginicaPetal))
            .toArray();
    long[] sizes = {setosaPetal.length, versicolorPetal.length, virginicaPetal.length};
    printResult("One-way ANOVA", Datamunge.oneWayAnova(allPetal, sizes));
    printResult("Kruskal-Wallis", Datamunge.kruskalWallisTest(allPetal, sizes));

    System.out.println(
        "\n=================== Correlation: sepal length vs. petal length ===================");
    double[] sepalLength = column(iris, "Sepal.Length", null);
    double[] petalLength = column(iris, "Petal.Length", null);
    printResult("Pearson correlation", Datamunge.pearsonCorrelationTest(sepalLength, petalLength));
    printResult(
        "Spearman correlation", Datamunge.spearmanCorrelationTest(sepalLength, petalLength));

    System.out.println(
        "\n=================== F-test: petal length variance, setosa vs. virginica"
            + " ===================");
    printResult("F test", Datamunge.fTestVariance(setosaPetal, virginicaPetal));

    System.out.println(
        "\n=================== Normality: is sepal length normally distributed within setosa?"
            + " ===================");
    double[] setosaSepal = column(iris, "Sepal.Length", "setosa");
    printResult("Shapiro-Francia", Datamunge.shapiroFranciaTest(setosaSepal));
    printResult(
        "KS vs. fitted normal", Datamunge.ksTestOneSampleNormal(setosaSepal, 5.006, 0.3525));

    System.out.println(
        "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of"
            + " species? ===================");
    double[] sortedPetal = allPetal.clone();
    Arrays.sort(sortedPetal);
    double medianAll = sortedPetal[sortedPetal.length / 2];
    long setosaLong = countAbove(setosaPetal, medianAll);
    long setosaShort = setosaPetal.length - setosaLong;
    long versicolorLong = countAbove(versicolorPetal, medianAll);
    long versicolorShort = versicolorPetal.length - versicolorLong;
    System.out.println(
        "table: setosa=["
            + setosaLong
            + ","
            + setosaShort
            + "] versicolor=["
            + versicolorLong
            + ","
            + versicolorShort
            + "]");
    double[] table = {setosaLong, setosaShort, versicolorLong, versicolorShort};
    printResult("Chi-squared independence", Datamunge.chiSquaredTestIndependence(table, 2, 2));
    printResult(
        "Fisher's exact test",
        Datamunge.fisherExactTest2x2(setosaLong, setosaShort, versicolorLong, versicolorShort));

    System.out.println(
        "\n=================== Proportion / binomial: fraction of \"long\" petals overall"
            + " ===================");
    long longCount = countAbove(allPetal, medianAll);
    long n = allPetal.length;
    printResult(
        "One-sample proportion test (vs 0.5)",
        Datamunge.proportionTestOneSample(longCount, n, 0.5));
    printResult("Exact binomial test (vs 0.5)", Datamunge.binomialTest(longCount, n, 0.5));
  }
}
