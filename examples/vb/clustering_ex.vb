' CMD: just TARGET=clustering_ex run-vb
' Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    Private species As List(Of String)

    ' For each cluster, print its size and its majority species (negative labels are noise).
    Private Sub ReportPurity(clusterLabels As IEnumerable(Of Long), nClusters As Integer)
        Dim votes = Enumerable.Range(0, nClusters).Select(Function(unused) New Dictionary(Of String, Integer)()).ToArray()
        For Each pair In clusterLabels.Zip(species)
            Dim label = pair.Item1, sp = pair.Item2
            If label >= 0 Then
                votes(CInt(label))(sp) = votes(CInt(label)).GetValueOrDefault(sp) + 1
            End If
        Next
        For c = 0 To nClusters - 1
            Dim total = votes(c).Values.Sum()
            If total = 0 Then Continue For
            Dim best = votes(c).MaxBy(Function(kv) kv.Value)
            Console.WriteLine($"  cluster {c}: {total} points, majority {best.Key} ({best.Value}/{total})")
        Next
    End Sub

    Sub Main()
        Dim iris = Data.Iris()
        species = Enumerable.Range(0, CInt(iris.Nrows())).Select(Function(i) iris.StringAt("Species", CULng(i))).ToList()
        Dim featureColumns = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}

        Console.WriteLine("=================== K-means (k=3) on iris ===================")
        Dim kmeans = New KMeans(iris, featureColumns, 3UL)
        Console.WriteLine($"Inertia: {kmeans.Inertia()}, iterations (best run): {kmeans.IterationsUsed()}")
        ReportPurity(kmeans.Labels().Select(Function(l) CLng(l)), 3)

        Console.WriteLine($"{vbLf}=================== Agglomerative clustering on iris ===================")
        For Each linkage In {"ward", "complete", "average"}
            Dim model = New AgglomerativeClustering(iris, featureColumns, 3UL, linkage)
            Console.WriteLine($"--- linkage={linkage} ---")
            ReportPurity(model.Labels().Select(Function(l) CLng(l)), 3)
        Next

        Console.WriteLine($"{vbLf}--- Re-cutting the ward dendrogram at k=2 without refitting ---")
        Dim wardModel = New AgglomerativeClustering(iris, featureColumns, 3UL, "ward")
        ReportPurity(wardModel.Cut(2UL).Select(Function(l) CLng(l)), 2)

        Console.WriteLine($"{vbLf}=================== DBSCAN on iris ===================")
        Dim dbscan = New DBSCAN(iris, featureColumns, 0.6, 5UL)
        Console.WriteLine($"Clusters found: {dbscan.NClusters()}, noise points: {dbscan.NNoise()} (of {dbscan.Observations()})")
        ReportPurity(dbscan.Labels().Select(Function(l) CLng(l)), CInt(dbscan.NClusters()))
    End Sub
End Module
