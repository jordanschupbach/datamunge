1;

datamunge;

% Octave's SWIG bindings use flat ClassName_method(obj, ...) function calls (no obj.method()
% sugar), and plain Octave arrays/cell arrays never auto-convert to a vector<T> parameter --
% build a real SVector via the object constructor + push_back.
function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

iris = DataFrame_iris();
features = sv({"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});

% species labels as an SVector, for the grouped (colored-by-species) scores plot
n = DataFrame_nrows(iris);
species_t = {};
for i = 0:(n - 1)
  species_t{end + 1} = DataFrame_string_at(iris, "Species", i);
end
species = sv(species_t);

printf("=================== PCA on iris (scaled) ===================\n");
pca = PCA(iris, features, true, true);
PCA_print_summary(pca);

printf("\nPC1 loadings (which original features drive it):\n");
% Returned std::vector<T> values arrive as native Octave cell arrays (1-based {i}); only
% *argument* vectors must be constructed SWIG objects (see sv() above).
names = PCA_feature_names(pca);
loadings = PCA_component_loadings(pca, 0);
for i = 1:numel(names)
  printf("  %s: %.4f\n", names{i}, loadings{i});
end

printf("\nTraining scores as a DataFrame:\n");
printf("%s\n", DataFrame_to_string(PCA_scores_frame(pca)));

scatter = PCA_plot_scores_grouped(pca, species);
Plot_save_svg(scatter, "pca_iris_scores.svg");
printf("\nScores scatter (colored by species) saved as pca_iris_scores.svg\n");

scree = PCA_plot_scree(pca);
Plot_save_svg(scree, "pca_iris_scree.svg");
printf("Scree plot saved as pca_iris_scree.svg\n");

printf("\n=================== PCA on iris (unscaled) ===================\n");
unscaled = PCA(iris, features, true, false);
scaled_ratio = PCA_explained_variance_ratio(pca);
unscaled_ratio = PCA_explained_variance_ratio(unscaled);
printf(["PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance " ...
        "dominates the unscaled covariance matrix.\n"], ...
       unscaled_ratio{1} * 100, scaled_ratio{1} * 100);
