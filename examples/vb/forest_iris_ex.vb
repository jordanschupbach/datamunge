' CMD: just TARGET=forest_iris_ex run-vb
' Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Dim model = New RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

        model.PlotClassification(iris, "Petal.Length", "Petal.Width").Save("forest_iris_classification.svg")
        model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("forest_iris_decision_regions.svg")
        Console.WriteLine($"{vbLf}Saved forest_iris_classification.svg and forest_iris_decision_regions.svg")

        ' A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
        ' are added -- the defining random forest effect a single decision tree cannot demonstrate.
        ' New RandomForestClassifier(data, formula, nTrees, maxDepth, minSamplesSplit, minSamplesLeaf,
        '                            maxFeatures, criterion, bootstrap, sampleFraction, seed)
        Dim smallForest = New RandomForestClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5UL)
        Console.WriteLine($"{vbLf}5-tree forest:   training accuracy={smallForest.TrainingAccuracy() * 100}%  OOB accuracy={smallForest.OobAccuracy() * 100}%")
        Console.WriteLine($"100-tree forest: training accuracy={model.TrainingAccuracy() * 100}%  OOB accuracy={model.OobAccuracy() * 100}%")
        smallForest.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("forest_iris_decision_regions_5trees.svg")
        Console.WriteLine("Saved forest_iris_decision_regions_5trees.svg")
    End Sub
End Module
