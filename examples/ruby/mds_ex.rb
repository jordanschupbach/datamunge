require "octruby"

iris = Datamunge::DataFrame.iris
features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
species = (0...iris.nrows).map { |i| iris.string_at("Species", i) }

puts "=================== Classical MDS on iris (euclidean) ==================="
mds = Datamunge::MDS.new(iris, features, 2, "euclidean")
mds.print_summary

puts "\nEmbedding as a DataFrame:"
puts mds.embedding_frame.to_string

scatter = mds.plot_embedding_grouped(species)
scatter.save_svg("mds_iris_embedding.svg")
puts "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg"

puts "\n=================== Classical MDS on iris (manhattan) ==================="
manhattan_mds = Datamunge::MDS.new(iris, features, 2, "manhattan")
puts format("Goodness of fit: euclidean=%.4f, manhattan=%.4f",
            mds.goodness_of_fit, manhattan_mds.goodness_of_fit)
