% CMD: just TARGET=pca_ex run-octave
% PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.

iris = datamunge.Data.iris ();
feature_columns = {'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'};

% species labels, for the grouped (colored-by-species) scores plot
species = iris.pull_string ('Species');

printf ('=================== PCA on iris (scaled) ===================\n');
pca = datamunge.PCA (iris, feature_columns, true, true);
pca.print_summary ();

printf ('\nPC1 loadings (which original features drive it):\n');
names = pca.feature_names ();
loadings = pca.component_loadings (0);
for i = 1:numel (names)
  printf ('  %s: %.4f\n', names{i}, loadings(i));
end

printf ('\nTraining scores as a DataFrame:\n%s\n', pca.scores_frame ().to_string ());

pca.plot_scores_grouped (species).save_svg ('pca_iris_scores.svg');
printf ('\nScores scatter (colored by species) saved as pca_iris_scores.svg\n');

pca.plot_scree ().save_svg ('pca_iris_scree.svg');
printf ('Scree plot saved as pca_iris_scree.svg\n');

printf ('\n=================== PCA on iris (unscaled) ===================\n');
unscaled = datamunge.PCA (iris, feature_columns, true, false);
printf (['PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length''s larger raw variance ' ...
         'dominates the unscaled covariance matrix.\n'], ...
        100 * unscaled.explained_variance_ratio ()(1), 100 * pca.explained_variance_ratio ()(1));
