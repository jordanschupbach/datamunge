' CMD: just TARGET=hypothesis_tests_ex run-vb
' Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    Private Sub PrintResult(label As String, r As HypothesisTestResult)
        Dim alt As String
        Select Case r.Alternative
            Case Alternative.Less : alt = "less"
            Case Alternative.Greater : alt = "greater"
            Case Else : alt = "two.sided"
        End Select
        Console.Write($"{label}: statistic={r.Statistic}")
        If r.Parameter1 <> 0 Then Console.Write($", df1={r.Parameter1}")
        If r.Parameter2 <> 0 Then Console.Write($", df2={r.Parameter2}")
        Console.Write($", p={r.PValue} ({alt})")
        If r.HasConfInt Then Console.Write($", CI=[{r.ConfIntLower}, {r.ConfIntUpper}]")
        Console.WriteLine($" -- {r.Method}")
    End Sub

    Private Function ColumnForSpecies(df As DataFrame, column As String, species As String) As List(Of Double)
        Dim values As New List(Of Double)
        For i As ULong = 0UL To df.Nrows() - 1UL
            If df.StringAt("Species", i) = species Then values.Add(df.NumericAt(column, i))
        Next
        Return values
    End Function

    Private Function Column(df As DataFrame, columnName As String) As List(Of Double)
        Dim values As New List(Of Double)
        For i As ULong = 0UL To df.Nrows() - 1UL
            values.Add(df.NumericAt(columnName, i))
        Next
        Return values
    End Function

    Sub Main()
        Dim iris = Data.Iris()

        Dim setosaPetal = ColumnForSpecies(iris, "Petal.Length", "setosa")
        Dim versicolorPetal = ColumnForSpecies(iris, "Petal.Length", "versicolor")
        Dim virginicaPetal = ColumnForSpecies(iris, "Petal.Length", "virginica")

        Console.WriteLine("=================== t-tests: petal length, setosa vs. versicolor ===================")
        PrintResult("Welch two-sample t-test", Functions.TTestTwoSample(setosaPetal, versicolorPetal))
        PrintResult("Wilcoxon rank-sum test", Functions.WilcoxonRankSumTest(setosaPetal, versicolorPetal))

        Console.WriteLine($"{vbLf}=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
        Dim allPetal = setosaPetal.Concat(versicolorPetal).Concat(virginicaPetal).ToList()
        Dim sizes = {CULng(setosaPetal.Count), CULng(versicolorPetal.Count), CULng(virginicaPetal.Count)}
        PrintResult("One-way ANOVA", Functions.OneWayAnova(allPetal, sizes))
        PrintResult("Kruskal-Wallis", Functions.KruskalWallisTest(allPetal, sizes))

        Console.WriteLine($"{vbLf}=================== Correlation: sepal length vs. petal length ===================")
        Dim sepalLength = Column(iris, "Sepal.Length")
        Dim petalLength = Column(iris, "Petal.Length")
        PrintResult("Pearson correlation", Functions.PearsonCorrelationTest(sepalLength, petalLength))
        PrintResult("Spearman correlation", Functions.SpearmanCorrelationTest(sepalLength, petalLength))

        Console.WriteLine($"{vbLf}=================== F-test: petal length variance, setosa vs. virginica ===================")
        PrintResult("F test", Functions.FTestVariance(setosaPetal, virginicaPetal))

        Console.WriteLine($"{vbLf}=================== Normality: is sepal length normally distributed within setosa? ===================")
        Dim setosaSepal = ColumnForSpecies(iris, "Sepal.Length", "setosa")
        PrintResult("Shapiro-Francia", Functions.ShapiroFranciaTest(setosaSepal))
        PrintResult("KS vs. fitted normal", Functions.KsTestOneSampleNormal(setosaSepal, 5.006, 0.3525))

        Console.WriteLine($"{vbLf}=================== Chi-squared / Fisher: is petal length ""long"" independent of species? ===================")
        Dim medianAll = allPetal.Order().ElementAt(allPetal.Count \ 2)
        Dim countLong = Function(v As IEnumerable(Of Double)) CULng(v.Count(Function(x) x > medianAll))
        Dim setosaLong = countLong(setosaPetal)
        Dim setosaShort = CULng(setosaPetal.Count) - setosaLong
        Dim versicolorLong = countLong(versicolorPetal)
        Dim versicolorShort = CULng(versicolorPetal.Count) - versicolorLong
        Console.WriteLine($"table: setosa=[{setosaLong},{setosaShort}] versicolor=[{versicolorLong},{versicolorShort}]")
        Dim table = {CDbl(setosaLong), CDbl(setosaShort), CDbl(versicolorLong), CDbl(versicolorShort)}
        PrintResult("Chi-squared independence", Functions.ChiSquaredTestIndependence(table, 2UL, 2UL))
        PrintResult("Fisher's exact test", Functions.FisherExactTest2x2(setosaLong, setosaShort, versicolorLong, versicolorShort))

        Console.WriteLine($"{vbLf}=================== Proportion / binomial: fraction of ""long"" petals overall ===================")
        Dim longCount = countLong(allPetal)
        Dim n = CULng(allPetal.Count)
        PrintResult("One-sample proportion test (vs 0.5)", Functions.ProportionTestOneSample(longCount, n, 0.5))
        PrintResult("Exact binomial test (vs 0.5)", Functions.BinomialTest(longCount, n, 0.5))
    End Sub
End Module
