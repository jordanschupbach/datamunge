# CMD: just TARGET=pca_ex run-ruby
# PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
require "datamunge"

iris = Datamunge::Data.iris
feature_columns = %w[Sepal.Length Sepal.Width Petal.Length Petal.Width]

# species labels, for the grouped (colored-by-species) scores plot
species = Array.new(iris.nrows) { |i| iris.string_at("Species", i) }

puts "=================== PCA on iris (scaled) ==================="
pca = Datamunge::PCA.new(iris, feature_columns, true, true)
pca.print_summary

puts "\nPC1 loadings (which original features drive it):"
pca.feature_names.zip(pca.component_loadings(0)) { |name, loading| printf("  %s: %.4f\n", name, loading) }

puts "\nTraining scores as a DataFrame:\n#{pca.scores_frame}"

pca.plot_scores_grouped(species).save_svg("pca_iris_scores.svg")
puts "\nScores scatter (colored by species) saved as pca_iris_scores.svg"

pca.plot_scree.save_svg("pca_iris_scree.svg")
puts "Scree plot saved as pca_iris_scree.svg"

puts "\n=================== PCA on iris (unscaled) ==================="
unscaled = Datamunge::PCA.new(iris, feature_columns, true, false)
printf("PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance " \
       "dominates the unscaled covariance matrix.\n",
       100 * unscaled.explained_variance_ratio[0], 100 * pca.explained_variance_ratio[0])
