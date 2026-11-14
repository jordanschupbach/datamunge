% CMD: just TARGET=mds_ex run-octave
% Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.

iris = datamunge.Data.iris ();
feature_columns = {'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'};
species = cell (1, iris.nrows ());
for i = 1:numel (species)
  species{i} = iris.string_at ('Species', i - 1);
end

printf ('=================== Classical MDS on iris (euclidean) ===================\n');
mds = datamunge.MDS (iris, feature_columns, 2, 'euclidean');
mds.print_summary ();

printf ('\nEmbedding as a DataFrame:\n%s\n', mds.embedding_frame ().to_string ());

mds.plot_embedding_grouped (species).save_svg ('mds_iris_embedding.svg');
printf ('\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n');

printf ('\n=================== Classical MDS on iris (manhattan) ===================\n');
manhattan_mds = datamunge.MDS (iris, feature_columns, 2, 'manhattan');
printf ('Goodness of fit: euclidean=%.4f, manhattan=%.4f\n', mds.goodness_of_fit (), manhattan_mds.goodness_of_fit ());
