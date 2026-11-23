' CMD: just TARGET=tree_iris_ex run-vb
' Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Dim model = New DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width")
        model.PrintSummary()

        Console.WriteLine($"{vbLf}Confusion matrix (rows = actual, cols = predicted):")
        Console.WriteLine(model.ConfusionMatrix())

        Console.WriteLine($"{vbLf}Misclassified rows:")
        Dim predictions = model.Predict(iris)
        Dim n = iris.Nrows()
        Dim misclassified = 0
        For i As ULong = 0UL To n - 1UL
            Dim actual = iris.StringAt("Species", i), predicted = predictions(CInt(i))
            If predicted = actual Then Continue For
            misclassified += 1
            Console.WriteLine($"  row {i}: Petal.Length={iris.NumericAt("Petal.Length", i)} Petal.Width={iris.NumericAt("Petal.Width", i)}  actual={actual}  predicted={predicted}")
        Next
        Console.WriteLine($"{misclassified} of {n} misclassified ({100.0 * misclassified / n}%)")

        model.PlotClassification(iris, "Petal.Length", "Petal.Width").Save("tree_iris_classification.svg")
        model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("tree_iris_decision_regions.svg")
        Console.WriteLine($"{vbLf}Saved tree_iris_classification.svg and tree_iris_decision_regions.svg")

        ' A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
        Dim shallow = New DecisionTreeClassifier(iris, "Species ~ Petal.Length + Petal.Width", 2UL)
        Console.WriteLine($"{vbLf}Depth-2 tree training accuracy: {shallow.TrainingAccuracy() * 100}% ({shallow.LeafCount()} leaves)")
        shallow.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("tree_iris_decision_regions_depth2.svg")
        Console.WriteLine("Saved tree_iris_decision_regions_depth2.svg")
    End Sub
End Module
