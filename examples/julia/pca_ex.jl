# CMD: just TARGET=pca_ex run-julia

# PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
using Datamunge
using Printf

const iris = Datamunge.Data.iris()
const feature_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

# species labels, for the grouped (colored-by-species) scores plot
const species = [string_at(iris, "Species", i) for i in 0:nrows(iris)-1]

println("=================== PCA on iris (scaled) ===================")
const pca = PCA(iris, feature_columns, true, true)
print_summary(pca)

println("\nPC1 loadings (which original features drive it):")
for (name, loading) in zip(feature_names(pca), component_loadings(pca, 0))
    @printf("  %s: %.4f\n", name, loading)
end

println("\nTraining scores as a DataFrame:\n$(to_string(scores_frame(pca)))")

save_svg(plot_scores_grouped(pca, species), "pca_iris_scores.svg")
println("\nScores scatter (colored by species) saved as pca_iris_scores.svg")

save_svg(plot_scree(pca), "pca_iris_scree.svg")
println("Scree plot saved as pca_iris_scree.svg")

println("\n=================== PCA on iris (unscaled) ===================")
let unscaled = PCA(iris, feature_columns, true, false)
    @printf("PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n",
            100 * explained_variance_ratio(unscaled)[1], 100 * explained_variance_ratio(pca)[1])
end
