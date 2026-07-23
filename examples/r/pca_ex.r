# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# feature_columns is passed via encode_strings() -- see examples/r/clustering_ex.r for why
# (a vector<string> constructor overload is unreachable from R; the encoded-string constructor
# overload works fine and is unambiguous, so the bare PCA(...) dispatcher resolves to it).
# plot_scores_grouped's `group_labels` is ALSO a vector<string>, but this time on an overloaded
# (default-arg-expanded) method -- that specific combination is unreachable via the bare
# dispatcher, so it needs the explicit __SWIG_2 (both defaults) form instead.
library(datamunger)

encode_strings <- function(values) paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
species <- c()
for (i in 0:(n - 1)) species <- c(species, DataFrame_string_at(iris, "Species", i))
features <- encode_strings(c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))

cat("=================== PCA on iris (scaled) ===================\n")
pca <- PCA(iris, features, TRUE, TRUE)
PCA_print_summary(pca)

cat("\nPC1 loadings (which original features drive it):\n")
loadings <- PCA_component_loadings(pca, 0)
feature_names <- PCA_feature_names(pca)
for (j in seq_along(feature_names)) cat("  ", feature_names[j], ":", loadings[j], "\n")

cat("\nTraining scores as a DataFrame:\n")
scores_df <- PCA_scores_frame(pca)
cat(DataFrame_to_string(scores_df), "\n")

scatter <- PCA_plot_scores_grouped__SWIG_2(pca, species)
Plot_save_svg(scatter, "pca_iris_scores.svg")
cat("\nScores scatter (colored by species) saved as pca_iris_scores.svg\n")

scree <- PCA_plot_scree(pca)
Plot_save_svg(scree, "pca_iris_scree.svg")
cat("Scree plot saved as pca_iris_scree.svg\n")

cat("\n=================== PCA on iris (unscaled) ===================\n")
unscaled <- PCA(iris, features, TRUE, FALSE)
scaled_ratio <- PCA_explained_variance_ratio(pca)
unscaled_ratio <- PCA_explained_variance_ratio(unscaled)
cat("PC1 explains", unscaled_ratio[1] * 100, "% of variance (vs.", scaled_ratio[1] * 100,
    "% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n")
