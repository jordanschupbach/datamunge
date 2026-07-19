module app;

import std.stdio : writeln;
import datamunge;

SVector sv(string[] t) {
  auto v = new SVector();
  foreach (x; t) v.push_back(x);
  return v;
}

int[] toIntArraySize(SizeVector v) {
  int[] out_;
  for (size_t i = 0; i < v.size(); i++) out_ ~= cast(int) v[i];
  return out_;
}

int[] toIntArrayI(IVector v) {
  int[] out_;
  for (size_t i = 0; i < v.size(); i++) out_ ~= v[i];
  return out_;
}

void report_purity(int[] cluster_labels, string[] species, size_t n_clusters) {
  int[string][] votes;
  votes.length = n_clusters;
  for (size_t i = 0; i < cluster_labels.length; i++) {
    auto label = cluster_labels[i];
    if (label < 0) continue;
    votes[label][species[i]] = (species[i] in votes[label] ? votes[label][species[i]] : 0) + 1;
  }
  for (size_t c = 0; c < n_clusters; c++) {
    int total = 0;
    foreach (v; votes[c]) total += v;
    if (total == 0) continue;
    string best_species;
    int best = -1;
    foreach (k, v; votes[c]) {
      if (v > best) { best = v; best_species = k; }
    }
    writeln("  cluster ", c, ": ", total, " points, majority ", best_species, " (", best, "/", total, ")");
  }
}

void main() {
  auto iris = DataFrame.iris();
  auto n = iris.nrows();
  string[] species;
  for (size_t i = 0; i < n; i++) species ~= iris.string_at("Species", i);
  auto features = sv(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]);

  writeln("=================== K-means (k=3) on iris ===================");
  auto kmeans = new KMeans(iris, features, 3);
  writeln("Inertia: ", kmeans.inertia(), ", iterations (best run): ", kmeans.iterations_used());
  report_purity(toIntArraySize(kmeans.labels()), species, 3);

  writeln("\n=================== Agglomerative clustering on iris ===================");
  foreach (linkage; ["ward", "complete", "average"]) {
    auto model = new AgglomerativeClustering(iris, features, 3, linkage);
    writeln("--- linkage=", linkage, " ---");
    report_purity(toIntArraySize(model.labels()), species, 3);
  }

  writeln("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
  auto ward_model = new AgglomerativeClustering(iris, features, 3, "ward");
  report_purity(toIntArraySize(ward_model.cut(2)), species, 2);

  writeln("\n=================== DBSCAN on iris ===================");
  auto dbscan = new DBSCAN(iris, features, 0.6, 5);
  writeln("Clusters found: ", dbscan.n_clusters(), ", noise points: ", dbscan.n_noise(), " (of ", dbscan.observations(), ")");
  report_purity(toIntArrayI(dbscan.labels()), species, dbscan.n_clusters());
}
