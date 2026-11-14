# CMD: just TARGET=mds_ex run-ruby
# Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
require "datamunge"

iris = Datamunge::Data.iris
feature_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
species = Array.new(iris.nrows) { |i| iris.string_at("Species", i) }

puts "=================== Classical MDS on iris (euclidean) ==================="
mds = Datamunge::MDS.new(iris, feature_columns, 2, "euclidean")
mds.print_summary

puts "\nEmbedding as a DataFrame:\n#{mds.embedding_frame}"

mds.plot_embedding_grouped(species).save_svg("mds_iris_embedding.svg")
puts "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg"

puts "\n=================== Classical MDS on iris (manhattan) ==================="
manhattan_mds = Datamunge::MDS.new(iris, feature_columns, 2, "manhattan")
printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n", mds.goodness_of_fit, manhattan_mds.goodness_of_fit)
