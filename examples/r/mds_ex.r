# CMD: nix develop .#r --command Rscript examples/r/mds_ex.r
library(datamunger)


iris <- DataFrame$iris()
n <- iris$nrows()
species <- c()
for (i in 0:(n - 1)) species <- c(species, iris$string_at("Species", i))
features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

cat("=================== Classical MDS on iris (euclidean) ===================\n")
mds <- MDS$new(iris, features, 2, "euclidean")
mds$print_summary()

cat("\nEmbedding as a DataFrame:\n")
embedding_df <- mds$embedding_frame()
cat(embedding_df$to_string(), "\n")

scatter <- mds$plot_embedding_grouped(species)
scatter$save_svg("mds_iris_embedding.svg")
cat("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n")

cat("\n=================== Classical MDS on iris (manhattan) ===================\n")
manhattan_mds <- MDS$new(iris, features, 2, "manhattan")
cat("Goodness of fit: euclidean=", mds$goodness_of_fit(), ", manhattan=", manhattan_mds$goodness_of_fit(), "\n")
