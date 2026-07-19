# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: feature_columns is passed via the *_encoded constructor (encode_strings()) since
# KMeans/AgglomerativeClustering/DBSCAN's vector<string> constructor overload is unreachable
# from R (a genuine C++-level SWIG coercion bug for any vector<string> parameter on an
# overloaded constructor -- see datamunge_r_dollar_dispatch_bug.md). This session added the
# encoded overload to the library specifically to work around it.
library(datamunger)

encode_strings <- function(values) paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))

report_purity <- function(cluster_labels, species, n_clusters) {
  for (c in 0:(n_clusters - 1)) {
    members <- species[cluster_labels == c]
    if (length(members) == 0) next
    tbl <- table(members)
    best <- names(tbl)[which.max(tbl)]
    cat("  cluster", c, ":", length(members), "points, majority", best, "(", max(tbl), "/", length(members), ")\n")
  }
}

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
species <- c()
for (i in 0:(n - 1)) species <- c(species, DataFrame_string_at(iris, "Species", i))
features <- encode_strings(c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))

cat("=================== K-means (k=3) on iris ===================\n")
kmeans_model <- KMeans(iris, features, 3)
cat("Inertia:", KMeans_inertia(kmeans_model), ", iterations (best run):", KMeans_iterations_used(kmeans_model), "\n")
report_purity(KMeans_labels(kmeans_model), species, 3)

cat("\n=================== Agglomerative clustering on iris ===================\n")
for (linkage in c("ward", "complete", "average")) {
  model <- AgglomerativeClustering(iris, features, 3, linkage)
  cat("--- linkage=", linkage, " ---\n")
  report_purity(AgglomerativeClustering_labels(model), species, 3)
}

cat("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n")
ward_model <- AgglomerativeClustering(iris, features, 3, "ward")
report_purity(AgglomerativeClustering_cut(ward_model, 2), species, 2)

cat("\n=================== DBSCAN on iris ===================\n")
dbscan_model <- DBSCAN(iris, features, 0.6, 5)
cat("Clusters found:", DBSCAN_n_clusters(dbscan_model), ", noise points:", DBSCAN_n_noise(dbscan_model),
    "(of", DBSCAN_observations(dbscan_model), ")\n")
report_purity(DBSCAN_labels(dbscan_model), species, DBSCAN_n_clusters(dbscan_model))
