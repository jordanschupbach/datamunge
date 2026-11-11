# CMD: nix develop .#r --command Rscript examples/r/clustering_ex.r
library(datamunger)


report_purity <- function(cluster_labels, species, n_clusters) {
  for (c in 0:(n_clusters - 1)) {
    members <- species[cluster_labels == c]
    if (length(members) == 0) next
    tbl <- table(members)
    best <- names(tbl)[which.max(tbl)]
    cat("  cluster", c, ":", length(members), "points, majority", best, "(", max(tbl), "/", length(members), ")\n")
  }
}

iris <- DataFrame$iris()
n <- iris$nrows()
species <- c()
for (i in 0:(n - 1)) species <- c(species, iris$string_at("Species", i))
features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

cat("=================== K-means (k=3) on iris ===================\n")
kmeans_model <- KMeans$new(iris, features, 3)
cat("Inertia:", kmeans_model$inertia(), ", iterations (best run):", kmeans_model$iterations_used(), "\n")
report_purity(kmeans_model$labels(), species, 3)

cat("\n=================== Agglomerative clustering on iris ===================\n")
for (linkage in c("ward", "complete", "average")) {
  model <- AgglomerativeClustering$new(iris, features, 3, linkage)
  cat("--- linkage=", linkage, " ---\n")
  report_purity(model$labels(), species, 3)
}

cat("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n")
ward_model <- AgglomerativeClustering$new(iris, features, 3, "ward")
report_purity(ward_model$cut(2), species, 2)

cat("\n=================== DBSCAN on iris ===================\n")
dbscan_model <- DBSCAN$new(iris, features, 0.6, 5)
cat("Clusters found:", dbscan_model$n_clusters(), ", noise points:", dbscan_model$n_noise(),
    "(of", dbscan_model$observations(), ")\n")
report_purity(dbscan_model$labels(), species, dbscan_model$n_clusters())
