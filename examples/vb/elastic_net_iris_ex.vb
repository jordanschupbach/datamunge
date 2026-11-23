' CMD: just TARGET=elastic_net_iris_ex run-vb
' Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Const formula As String = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols")
        Console.WriteLine($"formula: {formula}{vbLf}")

        Console.WriteLine("=================== Ridge ===================")
        Dim ridge = New Ridge(iris, formula)
        ridge.PrintSummary()

        Console.WriteLine($"{vbLf}=================== Lasso ===================")
        Dim lasso = New Lasso(iris, formula)
        lasso.PrintSummary()

        Console.WriteLine($"{vbLf}================= Elastic Net =================")
        ' New ElasticNet(data, formula, alpha=0.5, lambda, nLambda, cvFolds, standardize, seed)
        Dim elastic = New ElasticNet(iris, formula, 0.5)
        elastic.PrintSummary()

        Console.WriteLine($"{vbLf}Saved figures showing how each model's coefficients respond to the regularization strength, " &
                          "and the cross-validation curve used to pick it:")

        Dim figures As (Path As RPlot, Cv As RPlot, Name As String, Label As String)() = {
            (ridge.PlotCoefficientPath(), ridge.PlotCvCurve(), "ridge", "ridge:      "),
            (lasso.PlotCoefficientPath(), lasso.PlotCvCurve(), "lasso", "lasso:      "),
            (elastic.PlotCoefficientPath(), elastic.PlotCvCurve(), "elasticnet", "elastic net:")
        }
        For Each figure In figures
            figure.Path.Save($"elastic_net_{figure.Name}_path.svg")
            figure.Cv.Save($"elastic_net_{figure.Name}_cv.svg")
            Console.WriteLine($"  {figure.Label} elastic_net_{figure.Name}_path.svg, elastic_net_{figure.Name}_cv.svg")
        Next
    End Sub
End Module
