from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
species = [iris.string_at("Species", i) for i in range(iris.nrows())]

print("=================== PCA on iris (scaled) ===================")
pca = dm.PCA(iris, features, True, True)
pca.print_summary()

print("\nPC1 loadings (which original features drive it):")
for name, loading in zip(pca.feature_names(), pca.component_loadings(0)):
    print(f"  {name}: {loading:.4f}")

print("\nTraining scores as a DataFrame:")
scores_df = pca.scores_frame()
print(scores_df.to_string())

scatter = pca.plot_scores_grouped(species)
scatter.save_svg("pca_iris_scores.svg")
print("\nScores scatter (colored by species) saved as pca_iris_scores.svg")

scree = pca.plot_scree()
scree.save_svg("pca_iris_scree.svg")
print("Scree plot saved as pca_iris_scree.svg")

print("\n=================== PCA on iris (unscaled) ===================")
unscaled = dm.PCA(iris, features, True, False)
scaled_ratio = list(pca.explained_variance_ratio())
unscaled_ratio = list(unscaled.explained_variance_ratio())
print(
    f"PC1 explains {unscaled_ratio[0] * 100:.2f}% of variance "
    f"(vs. {scaled_ratio[0] * 100:.2f}% scaled) -- Sepal.Length's larger raw variance "
    "dominates the unscaled covariance matrix."
)
