' CMD: just TARGET=glm_iris_ex run-vb
' Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Dim rows = Enumerable.Range(0, CInt(iris.Nrows())).Select(Function(i) CULng(i)).ToList()

        ' Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
        ' the other two on these predictors, which sends logistic regression's coefficients toward
        ' +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
        ' binary split for a demo.
        Dim kept = rows.Where(Function(i) iris.StringAt("Species", i) = "versicolor" OrElse iris.StringAt("Species", i) = "virginica").ToList()
        Dim isVirginica = kept.Select(Function(i) If(iris.StringAt("Species", i) = "virginica", 1.0, 0.0)).ToList()
        Dim petalLength = kept.Select(Function(i) iris.NumericAt("Petal.Length", i)).ToList()
        Dim petalWidth = kept.Select(Function(i) iris.NumericAt("Petal.Width", i)).ToList()

        Dim [sub] = New DataFrame()
        [sub].AddNumericColumn("Petal.Length", petalLength)
        [sub].AddNumericColumn("Petal.Width", petalWidth)
        [sub].AddNumericColumn("is_virginica", isVirginica)

        Console.WriteLine("=================== Logistic regression (binomial, logit link) ===================")
        Dim logit = New GLM([sub], "is_virginica ~ Petal.Length + Petal.Width", "binomial")
        logit.PrintSummary()

        Dim fitted = logit.FittedValues()
        Dim n = isVirginica.Count
        Dim correct = Enumerable.Range(0, n).Count(Function(i) (fitted(i) >= 0.5) = (isVirginica(i) >= 0.5))
        Console.WriteLine($"{vbLf}Resubstitution accuracy at 0.5 threshold: {100.0 * correct / n}%")

        logit.SaveDiagnosticPlots("glm_logistic_iris")
        Console.WriteLine($"{vbLf}Saved glm_logistic_iris_{{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}}.svg")

        ' Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
        ' classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
        Const gridN As Integer = 100
        Dim widthMean = petalWidth.Average()
        Dim plMin = petalLength.Min() - 0.3, plMax = petalLength.Max() + 0.3
        Dim grid = New DataFrame()
        grid.AddNumericColumn("Petal.Length", Enumerable.Range(0, gridN).Select(Function(i) plMin + (plMax - plMin) * i / (gridN - 1)))
        grid.AddNumericColumn("Petal.Width", Enumerable.Repeat(widthMean, gridN))
        Dim curveFrame = logit.PredictFrame(grid, "confidence")
        Console.WriteLine($"{vbLf}Predicted-probability curve (first 5 rows):")
        Console.WriteLine(curveFrame.ToString(5UL))

        ' Poisson regression, for contrast: same IRLS engine, different family/link.
        Console.WriteLine($"{vbLf}=================== Poisson regression (log link) ===================")
        Dim countData = New DataFrame()
        countData.AddNumericColumn("Sepal.Width", rows.Select(Function(i) iris.NumericAt("Sepal.Width", i)))
        countData.AddNumericColumn("Petal.Length", rows.Select(Function(i) iris.NumericAt("Petal.Length", i)))
        countData.AddNumericColumn("count", rows.Select(Function(i) Math.Floor(iris.NumericAt("Sepal.Length", i) + 0.5)))
        Call New GLM(countData, "count ~ Sepal.Width + Petal.Length", "poisson").PrintSummary()
    End Sub
End Module
