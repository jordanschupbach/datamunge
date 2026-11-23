' CMD: just TARGET=gbm_iris_ex run-vb
' Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Dim model = New GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width")
        model.PrintSummary()

        Console.WriteLine($"{vbLf}Confusion matrix (rows = actual, cols = predicted):{vbLf}{model.ConfusionMatrix()}")

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

        model.PlotTrainingDeviance().Save("gbm_iris_training_deviance.svg")
        model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("gbm_iris_decision_regions.svg")
        Console.WriteLine($"{vbLf}Saved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg")

        ' A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
        ' at the training loss, gradually sharpening the decision boundary.
        ' New GBMClassifier(data, formula, nTrees, learningRate, maxDepth, minSamplesSplit,
        '                   minSamplesLeaf, subsample, seed)
        Dim few = New GBMClassifier(iris, "Species ~ Petal.Length + Petal.Width", 5UL)
        Console.WriteLine($"{vbLf}5-round ensemble:   training accuracy={100 * few.TrainingAccuracy()}%  deviance={few.TrainingDeviance().Last()}")
        Console.WriteLine($"100-round ensemble: training accuracy={100 * model.TrainingAccuracy()}%  deviance={model.TrainingDeviance().Last()}")
        few.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("gbm_iris_decision_regions_5rounds.svg")
        Console.WriteLine("Saved gbm_iris_decision_regions_5rounds.svg")
    End Sub
End Module
