# CMD: just TARGET=mds_ex run-julia

# Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
using Datamunge
using Printf

const iris = Datamunge.Data.iris()
const feature_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
const species = [string_at(iris, "Species", i) for i in 0:nrows(iris)-1]

println("=================== Classical MDS on iris (euclidean) ===================")
const mds = MDS(iris, feature_columns, 2, "euclidean")
print_summary(mds)

println("\nEmbedding as a DataFrame:\n$(to_string(embedding_frame(mds)))")

save_svg(plot_embedding_grouped(mds, species), "mds_iris_embedding.svg")
println("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg")

println("\n=================== Classical MDS on iris (manhattan) ===================")
let manhattan_mds = MDS(iris, feature_columns, 2, "manhattan")
    @printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n", goodness_of_fit(mds), goodness_of_fit(manhattan_mds))
end
