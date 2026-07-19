using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static void ReportPurity(IEnumerable<int> clusterLabelsEnum, List<string> species, int nClusters) {
    var clusterLabels = clusterLabelsEnum.ToList();
    var votes = new Dictionary<(int, string), int>();
    for (int i = 0; i < clusterLabels.Count; i++) {
      int label = clusterLabels[i];
      if (label >= 0) {
        var key = (label, species[i]);
        votes[key] = votes.GetValueOrDefault(key, 0) + 1;
      }
    }
    for (int c = 0; c < nClusters; c++) {
      int total = 0;
      int best = 0;
      string bestSp = "";
      foreach (var kv in votes) {
        if (kv.Key.Item1 == c) {
          total += kv.Value;
          if (kv.Value > best) {
            best = kv.Value;
            bestSp = kv.Key.Item2;
          }
        }
      }
      if (total > 0) {
        Console.WriteLine($"  cluster {c}: {total} points, majority {bestSp} ({best}/{total})");
      }
    }
  }

  static void Main() {
    var iris = DataFrame.iris();
    uint n = iris.nrows();
    var species = new List<string>();
    for (uint i = 0; i < n; i++) species.Add(iris.string_at("Species", i));
    var features = new SVector(new string[] { "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width" });

    Console.WriteLine("=================== K-means (k=3) on iris ===================");
    var kmeans = new KMeans(iris, features, 3);
    Console.WriteLine($"Inertia: {kmeans.inertia()}, iterations (best run): {kmeans.iterations_used()}");
    ReportPurity(kmeans.labels().Select(x => (int)x), species, 3);

    Console.WriteLine("\n=================== Agglomerative clustering on iris ===================");
    foreach (var linkage in new[] { "ward", "complete", "average" }) {
      var model = new AgglomerativeClustering(iris, features, 3, linkage);
      Console.WriteLine($"--- linkage={linkage} ---");
      ReportPurity(model.labels().Select(x => (int)x), species, 3);
    }

    Console.WriteLine("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
    var wardModel = new AgglomerativeClustering(iris, features, 3, "ward");
    ReportPurity(wardModel.cut(2).Select(x => (int)x), species, 2);

    Console.WriteLine("\n=================== DBSCAN on iris ===================");
    var dbscan = new DBSCAN(iris, features, 0.6, 5);
    Console.WriteLine($"Clusters found: {dbscan.n_clusters()}, noise points: {dbscan.n_noise()} (of {dbscan.observations()})");
    ReportPurity(dbscan.labels(), species, (int)dbscan.n_clusters());
  }
}
