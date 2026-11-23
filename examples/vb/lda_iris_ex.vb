' CMD: just TARGET=lda_iris_ex run-vb
' Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Dim model = New LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
        model.PrintSummary()

        Console.WriteLine($"{vbLf}Confusion matrix (rows = actual, cols = predicted):{vbLf}{model.ConfusionMatrix()}")

        Dim newdata = New DataFrame()
        newdata.AddNumericColumn("Sepal.Length", {5.1, 6.0, 6.5, 6.2})
        newdata.AddNumericColumn("Sepal.Width", {3.5, 2.7, 3.0, 2.8})
        newdata.AddNumericColumn("Petal.Length", {1.4, 4.5, 5.5, 4.8})
        newdata.AddNumericColumn("Petal.Width", {0.2, 1.5, 2.0, 1.8})
        Console.WriteLine($"{vbLf}Predictions for new flowers:{vbLf}{model.PredictFrame(newdata)}")

        model.SaveDiscriminantPlot("lda_iris_discriminants.svg")
        Console.WriteLine($"{vbLf}Saved discriminant plot as lda_iris_discriminants.svg")
    End Sub
End Module
