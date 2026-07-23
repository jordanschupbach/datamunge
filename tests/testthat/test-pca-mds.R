encode_strings <- function(values) paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))

test_that("PCA matches known iris variance-explained values and supports the full API", {
  iris_df <- DataFrame_iris()
  features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

  pca <- PCA__SWIG_3(iris_df, encode_strings(features), TRUE, TRUE)
  expect_equal(PCA_observations(pca), 150L)
  expect_equal(PCA_num_components(pca), 4L)

  ratio <- PCA_explained_variance_ratio(pca)
  expect_equal(round(ratio[1], 4), 0.7296)
  expect_equal(round(ratio[2], 4), 0.2285)
  expect_equal(sum(ratio), 1, tolerance = 1e-9)

  cumulative <- PCA_cumulative_explained_variance_ratio(pca)
  expect_equal(cumulative[length(cumulative)], 1, tolerance = 1e-9)

  scores_df <- PCA_scores_frame(pca)
  expect_equal(DataFrame_nrows(scores_df), 150L)
  expect_equal(DataFrame_columns(scores_df), c("PC1", "PC2", "PC3", "PC4"))

  species <- vapply(PCA_kept_row_indices(pca), function(i) DataFrame_string_at(iris_df, "Species", i), character(1))
  grouped_plot <- PCA_plot_scores_grouped__SWIG_2(pca, species)
  svg_path <- file.path(tempdir(), "test_pca_grouped.svg")
  Plot_save_svg(grouped_plot, svg_path)
  expect_true(file.exists(svg_path))

  scree <- PCA_plot_scree(pca)
  scree_svg_path <- file.path(tempdir(), "test_pca_scree.svg")
  Plot_save_svg(scree, scree_svg_path)
  expect_true(file.exists(scree_svg_path))
})

test_that("MDS embeds iris in 2D with high goodness of fit and supports the full API", {
  iris_df <- DataFrame_iris()
  features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

  mds <- MDS__SWIG_3(iris_df, encode_strings(features), 2L, "euclidean")
  expect_equal(MDS_observations(mds), 150L)
  expect_equal(MDS_n_components(mds), 2L)
  expect_gt(MDS_goodness_of_fit(mds), 0.9)

  embedding_df <- MDS_embedding_frame(mds)
  expect_equal(DataFrame_nrows(embedding_df), 150L)
  expect_equal(DataFrame_columns(embedding_df), c("Dim1", "Dim2"))

  species <- vapply(MDS_kept_row_indices(mds), function(i) DataFrame_string_at(iris_df, "Species", i), character(1))
  grouped_plot <- MDS_plot_embedding_grouped__SWIG_2(mds, species)
  svg_path <- file.path(tempdir(), "test_mds_grouped.svg")
  Plot_save_svg(grouped_plot, svg_path)
  expect_true(file.exists(svg_path))
})

test_that("PCA and MDS objects support $ method dispatch", {
  iris_df <- DataFrame_iris()
  features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

  pca <- PCA__SWIG_3(iris_df, encode_strings(features), TRUE, TRUE)
  expect_equal(pca$observations(), 150L)
  expect_equal(round(pca$explained_variance_ratio()[1], 4), 0.7296)

  mds <- MDS__SWIG_3(iris_df, encode_strings(features), 2L, "euclidean")
  expect_gt(mds$goodness_of_fit(), 0.9)
})
