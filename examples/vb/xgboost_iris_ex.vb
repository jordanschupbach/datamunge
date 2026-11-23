' CMD: just TARGET=xgboost_iris_ex run-vb
' XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Console.WriteLine($"iris: {iris.Nrows()} rows x {iris.Ncols()} cols{vbLf}")

        Dim model = New XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width")
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

        model.PlotTrainingDeviance().Save("xgboost_iris_training_deviance.svg")
        model.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("xgboost_iris_decision_regions.svg")
        Console.WriteLine($"{vbLf}Saved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg")

        ' XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
        ' grows the same deep trees but keeps every leaf weight small, producing a much smoother
        ' decision boundary than the lightly-regularized default.
        ' New XGBoostClassifier(data, formula, nTrees, learningRate, maxDepth, lambda, alpha, gamma,
        '                       minChildWeight, minSamplesLeaf, subsample, colsampleBytree, seed)
        Dim heavy = New XGBoostClassifier(iris, "Species ~ Petal.Length + Petal.Width", 100UL, 0.3, 6UL, 50.0)
        Console.WriteLine($"{vbLf}lambda=1 (default):   training accuracy={100 * model.TrainingAccuracy()}%  deviance={model.TrainingDeviance().Last()}")
        Console.WriteLine($"lambda=50 (heavy L2): training accuracy={100 * heavy.TrainingAccuracy()}%  deviance={heavy.TrainingDeviance().Last()}")
        heavy.PlotDecisionRegions("Petal.Length", "Petal.Width").Save("xgboost_iris_decision_regions_heavy_lambda.svg")
        Console.WriteLine("Saved xgboost_iris_decision_regions_heavy_lambda.svg")
    End Sub
End Module
