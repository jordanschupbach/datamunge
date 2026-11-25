// CMD: just TARGET=pca_ex run-javascript
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.

const datamunge = require("../../index.js");

const iris = datamunge.data.iris();
const features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

// species labels, for the grouped (colored-by-species) scores plot
const species = [];
for (let i = 0; i < iris.nrows(); i++) species.push(iris.string_at("Species", i));

console.log("=================== PCA on iris (scaled) ===================");
const pca = new datamunge.PCA(iris, features, true, true);
pca.print_summary();

console.log("\nPC1 loadings (which original features drive it):");
const names = pca.feature_names();
const loadings = pca.component_loadings(0);
for (let i = 0; i < names.length; i++) {
  console.log(`  ${names[i]}: ${loadings[i].toFixed(4)}`);
}

console.log("\nTraining scores as a DataFrame:");
console.log(pca.scores_frame().to_string());

pca.plot_scores_grouped(species).save_svg("pca_iris_scores.svg");
console.log("\nScores scatter (colored by species) saved as pca_iris_scores.svg");

pca.plot_scree().save_svg("pca_iris_scree.svg");
console.log("Scree plot saved as pca_iris_scree.svg");

console.log("\n=================== PCA on iris (unscaled) ===================");
const unscaled = new datamunge.PCA(iris, features, true, false);
const scaledRatio = pca.explained_variance_ratio();
const unscaledRatio = unscaled.explained_variance_ratio();
console.log(
  `PC1 explains ${(unscaledRatio[0] * 100).toFixed(2)}% of variance ` +
    `(vs. ${(scaledRatio[0] * 100).toFixed(2)}% scaled) -- Sepal.Length's larger raw variance ` +
    "dominates the unscaled covariance matrix."
);
