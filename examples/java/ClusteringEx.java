// CMD: just TARGET=clustering_ex run-java
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
import datamunge.AgglomerativeClustering;
import datamunge.DBSCAN;
import datamunge.KMeans;
import datamunge.data.Data;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ClusteringEx {
  static List<String> species = new ArrayList<>();

  // For each cluster, print its size and its majority species (negative labels are noise).
  static void reportPurity(long[] clusterLabels, long nClusters) {
    var votes = new LinkedHashMap<Long, LinkedHashMap<String, Integer>>();
    for (int i = 0; i < clusterLabels.length; i++) {
      long label = clusterLabels[i];
      if (label < 0) {
        continue;
      }
      votes
          .computeIfAbsent(label, k -> new LinkedHashMap<>())
          .merge(species.get(i), 1, Integer::sum);
    }
    for (long c = 0; c < nClusters; c++) {
      var counts = votes.get(c);
      if (counts == null) {
        continue;
      }
      int total = 0;
      String bestSpecies = "";
      int best = -1;
      for (Map.Entry<String, Integer> e : counts.entrySet()) {
        total += e.getValue();
        if (e.getValue() > best) {
          best = e.getValue();
          bestSpecies = e.getKey();
        }
      }
      System.out.println(
          "  cluster " + c + ": " + total + " points, majority " + bestSpecies + " (" + best + "/"
              + total + ")");
    }
  }

  static void reportPurity(int[] clusterLabels, long nClusters) {
    long[] widened = new long[clusterLabels.length];
    for (int i = 0; i < clusterLabels.length; i++) {
      widened[i] = clusterLabels[i];
    }
    reportPurity(widened, nClusters);
  }

  public static void main(String[] args) {
    var iris = Data.iris();
    for (int i = 0; i < iris.nrows(); i++) {
      species.add(iris.stringAt("Species", i));
    }
    var featureColumns = List.of("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");

    System.out.println("=================== K-means (k=3) on iris ===================");
    var kmeans = new KMeans(iris, featureColumns, 3);
    System.out.println(
        "Inertia: " + kmeans.inertia() + ", iterations (best run): " + kmeans.iterationsUsed());
    reportPurity(kmeans.labels(), 3);

    System.out.println(
        "\n=================== Agglomerative clustering on iris ===================");
    for (String linkage : List.of("ward", "complete", "average")) {
      try (var model = new AgglomerativeClustering(iris, featureColumns, 3, linkage)) {
        System.out.println("--- linkage=" + linkage + " ---");
        reportPurity(model.labels(), 3);
      }
    }

    System.out.println("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
    var wardModel = new AgglomerativeClustering(iris, featureColumns, 3, "ward");
    reportPurity(wardModel.cut(2), 2);

    System.out.println("\n=================== DBSCAN on iris ===================");
    var dbscan = new DBSCAN(iris, featureColumns, 0.6, 5);
    System.out.println(
        "Clusters found: " + dbscan.nClusters() + ", noise points: " + dbscan.nNoise() + " (of "
            + dbscan.observations() + ")");
    reportPurity(dbscan.labels(), dbscan.nClusters());
  }
}
