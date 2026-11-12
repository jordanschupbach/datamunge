# CMD: nix develop .#r --command Rscript examples/r/pca_ex.r
# REPL: nix develop .#r --command R --no-save --quiet
library(datamunger)


iris <- DataFrame$iris()
n <- iris$nrows()
species <- c()
for (i in 0:(n - 1)) species <- c(species, iris$string_at("Species", i))
features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

cat("=================== PCA on iris (scaled) ===================\n")
pca <- PCA$new(iris, features, TRUE, TRUE)
pca$print_summary()

cat("\nPC1 loadings (which original features drive it):\n")
loadings <- pca$component_loadings(0)
feature_names <- pca$feature_names()
for (j in seq_along(feature_names)) cat("  ", feature_names[j], ":", loadings[j], "\n")

cat("\nTraining scores as a DataFrame:\n")
scores_df <- pca$scores_frame()
cat(scores_df$to_string(), "\n")

scatter <- pca$plot_scores_grouped(species)
scatter$save_svg("pca_iris_scores.svg")
cat("\nScores scatter (colored by species) saved as pca_iris_scores.svg\n")

scree <- pca$plot_scree()
scree$save_svg("pca_iris_scree.svg")
cat("Scree plot saved as pca_iris_scree.svg\n")

cat("\n=================== PCA on iris (unscaled) ===================\n")
unscaled <- PCA$new(iris, features, TRUE, FALSE)
scaled_ratio <- pca$explained_variance_ratio()
unscaled_ratio <- unscaled$explained_variance_ratio()
cat("PC1 explains", unscaled_ratio[1] * 100, "% of variance (vs.", scaled_ratio[1] * 100,
    "% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n")
