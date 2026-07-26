module app;

import std.stdio : writeln, writefln;
import datamunge;

SVector sv(string[] t) {
  auto v = new SVector();
  foreach (x; t) v.push_back(x);
  return v;
}

void main() {
  auto iris = DataFrame.iris();
  auto features = sv(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]);

  // species labels as an SVector, for the grouped (colored-by-species) scores plot
  string[] species_t;
  for (size_t i = 0; i < iris.nrows(); i++) species_t ~= iris.string_at("Species", i);
  auto species = sv(species_t);

  writeln("=================== PCA on iris (scaled) ===================");
  auto pca = new PCA(iris, features, true, true);
  pca.print_summary();

  writeln("\nPC1 loadings (which original features drive it):");
  auto names = pca.feature_names();
  auto loadings = pca.component_loadings(0);
  for (size_t i = 0; i < names.size(); i++) {
    writefln("  %s: %.4f", names[i], loadings[i]);
  }

  writeln("\nTraining scores as a DataFrame:");
  writeln(pca.scores_frame().to_string());

  auto scatter = pca.plot_scores_grouped(species);
  scatter.save_svg("pca_iris_scores.svg");
  writeln("\nScores scatter (colored by species) saved as pca_iris_scores.svg");

  auto scree = pca.plot_scree();
  scree.save_svg("pca_iris_scree.svg");
  writeln("Scree plot saved as pca_iris_scree.svg");

  writeln("\n=================== PCA on iris (unscaled) ===================");
  auto unscaled = new PCA(iris, features, true, false);
  auto scaled_ratio = pca.explained_variance_ratio();
  auto unscaled_ratio = unscaled.explained_variance_ratio();
  writefln(
    "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance "
      ~ "dominates the unscaled covariance matrix.",
    unscaled_ratio[0] * 100, scaled_ratio[0] * 100);
}
