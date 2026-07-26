require "octruby"

# Plain Ruby arrays convert directly to std::vector<std::string> (feature/label lists).
iris = Datamunge::DataFrame.iris
features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
species = (0...iris.nrows).map { |i| iris.string_at("Species", i) }

puts "=================== PCA on iris (scaled) ==================="
pca = Datamunge::PCA.new(iris, features, true, true)
pca.print_summary

puts "\nPC1 loadings (which original features drive it):"
pca.feature_names.to_a.zip(pca.component_loadings(0).to_a).each do |name, loading|
  puts format("  %s: %.4f", name, loading)
end

puts "\nTraining scores as a DataFrame:"
puts pca.scores_frame.to_string

scatter = pca.plot_scores_grouped(species)
scatter.save_svg("pca_iris_scores.svg")
puts "\nScores scatter (colored by species) saved as pca_iris_scores.svg"

scree = pca.plot_scree
scree.save_svg("pca_iris_scree.svg")
puts "Scree plot saved as pca_iris_scree.svg"

puts "\n=================== PCA on iris (unscaled) ==================="
unscaled = Datamunge::PCA.new(iris, features, true, false)
scaled_ratio = pca.explained_variance_ratio.to_a
unscaled_ratio = unscaled.explained_variance_ratio.to_a
puts format(
  "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance " \
  "dominates the unscaled covariance matrix.",
  unscaled_ratio[0] * 100, scaled_ratio[0] * 100
)
