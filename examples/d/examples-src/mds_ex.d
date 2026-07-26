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

  string[] species_t;
  for (size_t i = 0; i < iris.nrows(); i++) species_t ~= iris.string_at("Species", i);
  auto species = sv(species_t);

  writeln("=================== Classical MDS on iris (euclidean) ===================");
  auto mds = new MDS(iris, features, 2, "euclidean");
  mds.print_summary();

  writeln("\nEmbedding as a DataFrame:");
  writeln(mds.embedding_frame().to_string());

  auto scatter = mds.plot_embedding_grouped(species);
  scatter.save_svg("mds_iris_embedding.svg");
  writeln("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg");

  writeln("\n=================== Classical MDS on iris (manhattan) ===================");
  auto manhattan_mds = new MDS(iris, features, 2, "manhattan");
  writefln("Goodness of fit: euclidean=%.4f, manhattan=%.4f",
    mds.goodness_of_fit(), manhattan_mds.goodness_of_fit());
}
