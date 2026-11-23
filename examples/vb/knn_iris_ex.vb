' CMD: just TARGET=knn_iris_ex run-vb
' k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Dim model = New KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width")
        model.PrintSummary()

        Console.WriteLine($"{vbLf}Confusion matrix (leave-one-out; rows = actual, cols = predicted):")
        Console.WriteLine(model.ConfusionMatrix())

        Console.WriteLine($"{vbLf}Leave-one-out misclassified rows:")
        Dim fitted = model.Predict(iris)
        Dim n = iris.Nrows()
        Dim misclassified = 0
        For i As ULong = 0UL To n - 1UL
            Dim actual = iris.StringAt("Species", i), predicted = fitted(CInt(i))
            If predicted = actual Then Continue For
            misclassified += 1
            Console.WriteLine($"  row {i}: Petal.Length={iris.NumericAt("Petal.Length", i)} Petal.Width={iris.NumericAt("Petal.Width", i)}  actual={actual}  predicted={predicted}")
        Next
        Console.WriteLine($"{misclassified} of {n} misclassified ({100.0 * misclassified / n}%)")

        model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("knn_iris_decision_regions_k5.svg")
        Console.WriteLine($"{vbLf}Saved knn_iris_decision_regions_k5.svg")

        ' k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
        ' every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
        ' underfit boundary). New KNNClassifier(data, formula, k, metric, weighted, standardize)
        For Each k In {1UL, 25UL}
            Dim knn = New KNNClassifier(iris, "Species ~ Petal.Length + Petal.Width", k)
            Dim file = $"knn_iris_decision_regions_k{k}.svg"
            Console.WriteLine($"{vbLf}k={k,-2} leave-one-out accuracy: {knn.TrainingAccuracy() * 100}%")
            knn.PlotDecisionRegions("Petal.Length", "Petal.Width").Save(file)
            Console.WriteLine($"Saved {file}")
        Next
    End Sub
End Module
