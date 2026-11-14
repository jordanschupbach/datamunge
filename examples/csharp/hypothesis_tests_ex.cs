// CMD: just TARGET=hypothesis_tests_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
using Datamunge;

static void PrintResult(string label, HypothesisTestResult r)
{
    var alternative = r.Alternative switch
    {
        Alternative.Less => "less",
        Alternative.Greater => "greater",
        _ => "two.sided",
    };
    Console.Write($"{label}: statistic={r.Statistic}");
    if (r.Parameter1 != 0) Console.Write($", df1={r.Parameter1}");
    if (r.Parameter2 != 0) Console.Write($", df2={r.Parameter2}");
    Console.Write($", p={r.PValue} ({alternative})");
    if (r.HasConfInt) Console.Write($", CI=[{r.ConfIntLower}, {r.ConfIntUpper}]");
    Console.WriteLine($" -- {r.Method}");
}

static List<double> ColumnForSpecies(DataFrame df, string column, string species)
{
    List<double> values = [];
    for (ulong i = 0; i < df.Nrows(); i++)
        if (df.StringAt("Species", i) == species)
            values.Add(df.NumericAt(column, i));
    return values;
}

static List<double> Column(DataFrame df, string column)
{
    List<double> values = [];
    for (ulong i = 0; i < df.Nrows(); i++)
        values.Add(df.NumericAt(column, i));
    return values;
}

var iris = Data.Iris()!;

var setosaPetal = ColumnForSpecies(iris, "Petal.Length", "setosa");
var versicolorPetal = ColumnForSpecies(iris, "Petal.Length", "versicolor");
var virginicaPetal = ColumnForSpecies(iris, "Petal.Length", "virginica");

Console.WriteLine("=================== t-tests: petal length, setosa vs. versicolor ===================");
PrintResult("Welch two-sample t-test", Functions.TTestTwoSample(setosaPetal, versicolorPetal));
PrintResult("Wilcoxon rank-sum test", Functions.WilcoxonRankSumTest(setosaPetal, versicolorPetal));

Console.WriteLine("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
List<double> allPetal = [.. setosaPetal, .. versicolorPetal, .. virginicaPetal];
ulong[] sizes = [(ulong)setosaPetal.Count, (ulong)versicolorPetal.Count, (ulong)virginicaPetal.Count];
PrintResult("One-way ANOVA", Functions.OneWayAnova(allPetal, sizes));
PrintResult("Kruskal-Wallis", Functions.KruskalWallisTest(allPetal, sizes));

Console.WriteLine("\n=================== Correlation: sepal length vs. petal length ===================");
var sepalLength = Column(iris, "Sepal.Length");
var petalLength = Column(iris, "Petal.Length");
PrintResult("Pearson correlation", Functions.PearsonCorrelationTest(sepalLength, petalLength));
PrintResult("Spearman correlation", Functions.SpearmanCorrelationTest(sepalLength, petalLength));

Console.WriteLine("\n=================== F-test: petal length variance, setosa vs. virginica ===================");
PrintResult("F test", Functions.FTestVariance(setosaPetal, virginicaPetal));

Console.WriteLine("\n=================== Normality: is sepal length normally distributed within setosa? ===================");
var setosaSepal = ColumnForSpecies(iris, "Sepal.Length", "setosa");
PrintResult("Shapiro-Francia", Functions.ShapiroFranciaTest(setosaSepal));
PrintResult("KS vs. fitted normal", Functions.KsTestOneSampleNormal(setosaSepal, 5.006, 0.3525));

Console.WriteLine("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================");
var medianAll = allPetal.Order().ElementAt(allPetal.Count / 2);
ulong CountLong(IEnumerable<double> v) => (ulong)v.Count(x => x > medianAll);
{
    var setosaLong = CountLong(setosaPetal);
    var setosaShort = (ulong)setosaPetal.Count - setosaLong;
    var versicolorLong = CountLong(versicolorPetal);
    var versicolorShort = (ulong)versicolorPetal.Count - versicolorLong;
    Console.WriteLine($"table: setosa=[{setosaLong},{setosaShort}] versicolor=[{versicolorLong},{versicolorShort}]");
    double[] table = [setosaLong, setosaShort, versicolorLong, versicolorShort];
    PrintResult("Chi-squared independence", Functions.ChiSquaredTestIndependence(table, 2, 2));
    PrintResult("Fisher's exact test", Functions.FisherExactTest2x2(setosaLong, setosaShort, versicolorLong, versicolorShort));
}

Console.WriteLine("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================");
{
    var longCount = CountLong(allPetal);
    var n = (ulong)allPetal.Count;
    PrintResult("One-sample proportion test (vs 0.5)", Functions.ProportionTestOneSample(longCount, n, 0.5));
    PrintResult("Exact binomial test (vs 0.5)", Functions.BinomialTest(longCount, n, 0.5));
}
