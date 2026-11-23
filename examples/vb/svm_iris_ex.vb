' CMD: just TARGET=svm_iris_ex run-vb
' Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Const formula As String = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"

    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Console.WriteLine("=== RBF kernel (default) ===")
        Dim rbfModel = New SVM(iris, formula)
        rbfModel.PrintSummary()
        Console.WriteLine($"{vbLf}Confusion matrix (rows = actual, cols = predicted):{vbLf}{rbfModel.ConfusionMatrix()}")

        Console.WriteLine($"{vbLf}=== Linear kernel, for comparison ===")
        Dim linearModel = New SVM(iris, formula, "linear")
        Console.WriteLine($"Training accuracy: {linearModel.TrainingAccuracy() * 100}%")
        Console.WriteLine($"Support vectors: {linearModel.NumSupportVectors()}")

        Dim newdata = New DataFrame()
        newdata.AddNumericColumn("Sepal.Length", {5.1, 6.0, 6.5, 6.2})
        newdata.AddNumericColumn("Sepal.Width", {3.5, 2.7, 3.0, 2.8})
        newdata.AddNumericColumn("Petal.Length", {1.4, 4.5, 5.5, 4.8})
        newdata.AddNumericColumn("Petal.Width", {0.2, 1.5, 2.0, 1.8})
        Console.WriteLine($"{vbLf}RBF predictions for new flowers (votes out of 3 one-vs-one pairs):{vbLf}{rbfModel.PredictFrame(newdata)}")
    End Sub
End Module
