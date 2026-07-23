# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# feature_columns is passed via encode_strings() -- see examples/r/clustering_ex.r for why.
# plot_embedding_grouped's `group_labels` is ALSO a vector<string>, but this time on an
# overloaded (default-arg-expanded) method -- that specific combination is unreachable via the
# bare dispatcher, so it needs the explicit __SWIG_2 (both defaults) form instead; see pca_ex.r.
library(datamunger)

encode_strings <- function(values) paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))

iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
species <- c()
for (i in 0:(n - 1)) species <- c(species, DataFrame_string_at(iris, "Species", i))
features <- encode_strings(c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))

cat("=================== Classical MDS on iris (euclidean) ===================\n")
mds <- MDS(iris, features, 2, "euclidean")
MDS_print_summary(mds)

cat("\nEmbedding as a DataFrame:\n")
embedding_df <- MDS_embedding_frame(mds)
cat(DataFrame_to_string(embedding_df), "\n")

scatter <- MDS_plot_embedding_grouped__SWIG_2(mds, species)
Plot_save_svg(scatter, "mds_iris_embedding.svg")
cat("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg\n")

cat("\n=================== Classical MDS on iris (manhattan) ===================\n")
manhattan_mds <- MDS(iris, features, 2, "manhattan")
cat("Goodness of fit: euclidean=", MDS_goodness_of_fit(mds), ", manhattan=", MDS_goodness_of_fit(manhattan_mds), "\n")
