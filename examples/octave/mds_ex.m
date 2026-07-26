1;

datamunge;

function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

iris = DataFrame_iris();
features = sv({"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});

n = DataFrame_nrows(iris);
species_t = {};
for i = 0:(n - 1)
  species_t{end + 1} = DataFrame_string_at(iris, "Species", i);
end
species = sv(species_t);

printf("=================== Classical MDS on iris (euclidean) ===================\n");
mds = MDS(iris, features, 2, "euclidean");
MDS_print_summary(mds);

printf("\nEmbedding as a DataFrame:\n");
printf("%s\n", DataFrame_to_string(MDS_embedding_frame(mds)));

scatter = MDS_plot_embedding_grouped(mds, species);
Plot_save_svg(scatter, "mds_iris_embedding.svg");
printf("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n");

printf("\n=================== Classical MDS on iris (manhattan) ===================\n");
manhattan_mds = MDS(iris, features, 2, "manhattan");
printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n", ...
       MDS_goodness_of_fit(mds), MDS_goodness_of_fit(manhattan_mds));
