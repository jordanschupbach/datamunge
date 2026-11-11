
test_that("PCA matches known iris variance-explained values and supports the full API", {
  iris_df <- DataFrame$iris()
  features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

  pca <- PCA$new(iris_df, features, TRUE, TRUE)
  expect_equal(pca$observations(), 150L)
  expect_equal(pca$num_components(), 4L)

  ratio <- pca$explained_variance_ratio()
  expect_equal(round(ratio[1], 4), 0.7296)
  expect_equal(round(ratio[2], 4), 0.2285)
  expect_equal(sum(ratio), 1, tolerance = 1e-9)

  cumulative <- pca$cumulative_explained_variance_ratio()
  expect_equal(cumulative[length(cumulative)], 1, tolerance = 1e-9)

  scores_df <- pca$scores_frame()
  expect_equal(scores_df$nrows(), 150L)
  expect_equal(scores_df$columns(), c("PC1", "PC2", "PC3", "PC4"))

  species <- vapply(pca$kept_row_indices(), function(i) iris_df$string_at("Species", i), character(1))
  grouped_plot <- pca$plot_scores_grouped(species)
  svg_path <- file.path(tempdir(), "test_pca_grouped.svg")
  grouped_plot$save_svg(svg_path)
  expect_true(file.exists(svg_path))

  scree <- pca$plot_scree()
  scree_svg_path <- file.path(tempdir(), "test_pca_scree.svg")
  scree$save_svg(scree_svg_path)
  expect_true(file.exists(scree_svg_path))
})

test_that("MDS embeds iris in 2D with high goodness of fit and supports the full API", {
  iris_df <- DataFrame$iris()
  features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

  mds <- MDS$new(iris_df, features, 2L, "euclidean")
  expect_equal(mds$observations(), 150L)
  expect_equal(mds$n_components(), 2L)
  expect_gt(mds$goodness_of_fit(), 0.9)

  embedding_df <- mds$embedding_frame()
  expect_equal(embedding_df$nrows(), 150L)
  expect_equal(embedding_df$columns(), c("Dim1", "Dim2"))

  species <- vapply(mds$kept_row_indices(), function(i) iris_df$string_at("Species", i), character(1))
  grouped_plot <- mds$plot_embedding_grouped(species)
  svg_path <- file.path(tempdir(), "test_mds_grouped.svg")
  grouped_plot$save_svg(svg_path)
  expect_true(file.exists(svg_path))
})

test_that("PCA and MDS objects support $ method dispatch", {
  iris_df <- DataFrame$iris()
  features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

  pca <- PCA$new(iris_df, features, TRUE, TRUE)
  expect_equal(pca$observations(), 150L)
  expect_equal(round(pca$explained_variance_ratio()[1], 4), 0.7296)

  mds <- MDS$new(iris_df, features, 2L, "euclidean")
  expect_gt(mds$goodness_of_fit(), 0.9)
})
