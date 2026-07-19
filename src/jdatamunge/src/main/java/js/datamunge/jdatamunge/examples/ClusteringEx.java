package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.SVector;
import js.datamunge.jdatamunge.SizeVector;
import js.datamunge.jdatamunge.IVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.KMeans;
import js.datamunge.jdatamunge.AgglomerativeClustering;
import js.datamunge.jdatamunge.DBSCAN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClusteringEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static void reportPurity(List<Long> labels, List<String> species, long nClusters) {
    Map<String, Integer> votes = new HashMap<>();
    for (int i = 0; i < labels.size(); i++) {
      long label = labels.get(i);
      if (label >= 0) {
        String key = label + "," + species.get(i);
        votes.merge(key, 1, Integer::sum);
      }
    }
    for (long c = 0; c < nClusters; c++) {
      int total = 0;
      int best = 0;
      String bestSp = "";
      for (Map.Entry<String, Integer> e : votes.entrySet()) {
        String[] parts = e.getKey().split(",", 2);
        if (Long.parseLong(parts[0]) == c) {
          total += e.getValue();
          if (e.getValue() > best) {
            best = e.getValue();
            bestSp = parts[1];
          }
        }
      }
      if (total > 0) {
        System.out.println("  cluster " + c + ": " + total + " points, majority " + bestSp + " (" + best + "/" + total + ")");
      }
    }
  }

  static List<Long> szvToList(SizeVector v) {
    List<Long> out = new ArrayList<>();
    for (int i = 0; i < v.size(); i++) out.add(v.get(i));
    return out;
  }

  static List<Long> ivToList(IVector v) {
    List<Long> out = new ArrayList<>();
    for (int i = 0; i < v.size(); i++) out.add((long) v.get(i));
    return out;
  }

  public static void run() {
    var iris = DataFrame.iris();
    long n = iris.nrows();
    List<String> species = new ArrayList<>();
    for (long i = 0; i < n; i++) species.add(iris.string_at("Species", i));
    var features = new SVector(new String[] {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});

    System.out.println("=================== K-means (k=3) on iris ===================");
    var kmeans = new KMeans(iris, features, 3);
    System.out.println("Inertia: " + kmeans.inertia() + ", iterations (best run): " + kmeans.iterations_used());
    reportPurity(szvToList(kmeans.labels()), species, 3);

    System.out.println("\n=================== Agglomerative clustering on iris ===================");
    for (String linkage : new String[] {"ward", "complete", "average"}) {
      var model = new AgglomerativeClustering(iris, features, 3, linkage);
      System.out.println("--- linkage=" + linkage + " ---");
      reportPurity(szvToList(model.labels()), species, 3);
    }

    System.out.println("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
    var wardModel = new AgglomerativeClustering(iris, features, 3, "ward");
    reportPurity(szvToList(wardModel.cut(2)), species, 2);

    System.out.println("\n=================== DBSCAN on iris ===================");
    var dbscan = new DBSCAN(iris, features, 0.6, 5);
    System.out.println("Clusters found: " + dbscan.n_clusters() + ", noise points: " + dbscan.n_noise() + " (of " + dbscan.observations() + ")");
    reportPurity(ivToList(dbscan.labels()), species, dbscan.n_clusters());
  }
}
