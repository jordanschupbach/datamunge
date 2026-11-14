// CMD: just TARGET=clustering_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
using Datamunge;

var iris = Data.Iris()!;
var species = Enumerable.Range(0, (int)iris.Nrows()).Select(i => iris.StringAt("Species", (ulong)i)).ToList();
string[] featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

// For each cluster, print its size and its majority species (negative labels are noise).
void ReportPurity(IEnumerable<long> clusterLabels, int nClusters)
{
    var votes = Enumerable.Range(0, nClusters).Select(_ => new Dictionary<string, int>()).ToArray();
    foreach (var (label, sp) in clusterLabels.Zip(species))
        if (label >= 0)
            votes[label][sp] = votes[label].GetValueOrDefault(sp) + 1;
    for (var c = 0; c < nClusters; c++)
    {
        var total = votes[c].Values.Sum();
        if (total == 0) continue;
        var (bestSpecies, best) = votes[c].MaxBy(kv => kv.Value);
        Console.WriteLine($"  cluster {c}: {total} points, majority {bestSpecies} ({best}/{total})");
    }
}

Console.WriteLine("=================== K-means (k=3) on iris ===================");
var kmeans = new KMeans(iris, featureColumns, 3);
Console.WriteLine($"Inertia: {kmeans.Inertia()}, iterations (best run): {kmeans.IterationsUsed()}");
ReportPurity(kmeans.Labels().Select(l => (long)l), 3);

Console.WriteLine("\n=================== Agglomerative clustering on iris ===================");
foreach (var linkage in new[] { "ward", "complete", "average" })
{
    var model = new AgglomerativeClustering(iris, featureColumns, 3, linkage);
    Console.WriteLine($"--- linkage={linkage} ---");
    ReportPurity(model.Labels().Select(l => (long)l), 3);
}

Console.WriteLine("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
var wardModel = new AgglomerativeClustering(iris, featureColumns, 3, "ward");
ReportPurity(wardModel.Cut(2).Select(l => (long)l), 2);

Console.WriteLine("\n=================== DBSCAN on iris ===================");
var dbscan = new DBSCAN(iris, featureColumns, 0.6, 5);
Console.WriteLine($"Clusters found: {dbscan.NClusters()}, noise points: {dbscan.NNoise()} (of {dbscan.Observations()})");
ReportPurity(dbscan.Labels().Select(l => (long)l), (int)dbscan.NClusters());
