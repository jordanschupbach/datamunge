' CMD: just TARGET=mds_ex run-vb
' Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Dim iris = Data.Iris()
        Dim featureColumns = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}
        Dim species = Enumerable.Range(0, CInt(iris.Nrows())).Select(Function(i) iris.StringAt("Species", CULng(i))).ToList()

        Console.WriteLine("=================== Classical MDS on iris (euclidean) ===================")
        Dim mds = New MDS(iris, featureColumns, 2UL, "euclidean")
        mds.PrintSummary()

        Console.WriteLine($"{vbLf}Embedding as a DataFrame:{vbLf}{mds.EmbeddingFrame()}")

        mds.PlotEmbeddingGrouped(species).SaveSvg("mds_iris_embedding.svg")
        Console.WriteLine($"{vbLf}Embedding scatter (colored by species) saved as mds_iris_embedding.svg")

        Console.WriteLine($"{vbLf}=================== Classical MDS on iris (manhattan) ===================")
        Dim manhattanMds = New MDS(iris, featureColumns, 2UL, "manhattan")
        Console.WriteLine($"Goodness of fit: euclidean={mds.GoodnessOfFit():F4}, manhattan={manhattanMds.GoodnessOfFit():F4}")
    End Sub
End Module
