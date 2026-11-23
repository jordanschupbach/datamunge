' CMD: just TARGET=pca_ex run-vb
' PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Dim featureColumns = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}

        ' species labels, for the grouped (colored-by-species) scores plot
        Dim species = Enumerable.Range(0, CInt(iris.Nrows())).Select(Function(i) iris.StringAt("Species", CULng(i))).ToList()

        Console.WriteLine("=================== PCA on iris (scaled) ===================")
        Dim pca = New PCA(iris, featureColumns, True, True)
        pca.PrintSummary()

        Console.WriteLine($"{vbLf}PC1 loadings (which original features drive it):")
        For Each pair In pca.FeatureNames().Zip(pca.ComponentLoadings(0UL))
            Console.WriteLine($"  {pair.Item1}: {pair.Item2:F4}")
        Next

        Console.WriteLine($"{vbLf}Training scores as a DataFrame:{vbLf}{pca.ScoresFrame()}")

        pca.PlotScoresGrouped(species).SaveSvg("pca_iris_scores.svg")
        Console.WriteLine($"{vbLf}Scores scatter (colored by species) saved as pca_iris_scores.svg")

        pca.PlotScree().SaveSvg("pca_iris_scree.svg")
        Console.WriteLine("Scree plot saved as pca_iris_scree.svg")

        Console.WriteLine($"{vbLf}=================== PCA on iris (unscaled) ===================")
        Dim unscaled = New PCA(iris, featureColumns, True, False)
        Console.WriteLine($"PC1 explains {100 * unscaled.ExplainedVarianceRatio()(0):F2}% of variance (vs. {100 * pca.ExplainedVarianceRatio()(0):F2}% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.")
    End Sub
End Module
