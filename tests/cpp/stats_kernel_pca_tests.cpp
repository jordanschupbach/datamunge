#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/kernel_pca.hpp>
#include <datamunge/stats/pca.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::KernelPCA;
using datamunge::stats::KernelPCAOptions;
using datamunge::stats::PCA;
using datamunge::stats::PCAOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

double embedded_distance(const datamunge::linalg::DenseMatrix<double>& emb, std::size_t i, std::size_t j) {
  double sum = 0.0;
  for (std::size_t c = 0; c < emb.cols(); ++c) {
    const double d = emb(i, c) - emb(j, c);
    sum += d * d;
  }
  return std::sqrt(sum);
}

double average_within_between_ratio(const datamunge::linalg::DenseMatrix<double>& emb, const DataFrame& iris,
                                    const std::vector<std::size_t>& kept) {
  std::vector<std::string> species(kept.size());
  for (std::size_t i = 0; i < kept.size(); ++i) species[i] = iris.string_at("Species", kept[i]);

  double within_sum = 0.0, between_sum = 0.0;
  std::size_t within_n = 0, between_n = 0;
  for (std::size_t i = 0; i < kept.size(); ++i) {
    for (std::size_t j = i + 1; j < kept.size(); ++j) {
      const double d = embedded_distance(emb, i, j);
      if (species[i] == species[j]) {
        within_sum += d;
        ++within_n;
      } else {
        between_sum += d;
        ++between_n;
      }
    }
  }
  return (between_sum / static_cast<double>(between_n)) / (within_sum / static_cast<double>(within_n));
}

} // namespace

TEST(KernelPCA, EmbeddingShapeAndFinitenessForAllKernels) {
  const auto iris = datamunge::datasets::iris();
  for (const std::string& kernel : {"linear", "rbf", "polynomial"}) {
    KernelPCAOptions opts;
    opts.n_components = 2;
    opts.kernel = kernel;
    opts.gamma = 0.1;
    KernelPCA kpca(iris, kIrisFeatures, opts);

    EXPECT_EQ(kpca.observations(), iris.nrows());
    EXPECT_EQ(kpca.n_components(), 2u);
    EXPECT_EQ(kpca.eigenvalues().size(), 2u);
    for (std::size_t i = 0; i < kpca.observations(); ++i)
      for (std::size_t c = 0; c < kpca.n_components(); ++c) EXPECT_TRUE(std::isfinite(kpca.embedding()(i, c)));
    EXPECT_GT(kpca.eigenvalues()[0], kpca.eigenvalues()[1]);
    EXPECT_GT(kpca.eigenvalues()[1], 0.0);
  }
}

TEST(KernelPCA, LinearKernelReproducesUnscaledPcaUpToSignFlip) {
  const auto iris = datamunge::datasets::iris();
  KernelPCA kpca(iris, kIrisFeatures, KernelPCAOptions{2, "linear", 1.0, 3.0, 1.0});
  PCA pca(iris, kIrisFeatures, PCAOptions{true, false});

  ASSERT_EQ(kpca.observations(), pca.observations());

  // Determine the sign flip from the first observation, then check it holds throughout.
  const double sign0 = (kpca.embedding()(0, 0) * pca.scores()(0, 0) >= 0.0) ? 1.0 : -1.0;
  const double sign1 = (kpca.embedding()(0, 1) * pca.scores()(0, 1) >= 0.0) ? 1.0 : -1.0;

  for (std::size_t i = 0; i < kpca.observations(); ++i) {
    EXPECT_NEAR(kpca.embedding()(i, 0), sign0 * pca.scores()(i, 0), 1e-6);
    EXPECT_NEAR(kpca.embedding()(i, 1), sign1 * pca.scores()(i, 1), 1e-6);
  }

  // Eigenvalues of the (unscaled) centered kernel matrix equal PCA's covariance eigenvalues
  // times (n - 1) -- the kernel matrix K = X X^T has the same nonzero eigenvalues as the p x p
  // covariance-like matrix X^T X, scaled by (n - 1) relative to the sample covariance.
  const double n_minus_1 = static_cast<double>(kpca.observations() - 1);
  EXPECT_NEAR(kpca.eigenvalues()[0], pca.explained_variance()[0] * n_minus_1, 1e-4);
  EXPECT_NEAR(kpca.eigenvalues()[1], pca.explained_variance()[1] * n_minus_1, 1e-4);
}

TEST(KernelPCA, RbfKernelSeparatesIrisSpeciesAtLeastAsWellAsLinear) {
  const auto iris = datamunge::datasets::iris();
  KernelPCA linear_kpca(iris, kIrisFeatures, KernelPCAOptions{2, "linear", 1.0, 3.0, 1.0});
  KernelPCA rbf_kpca(iris, kIrisFeatures, KernelPCAOptions{2, "rbf", 0.13, 3.0, 1.0});

  const double linear_ratio = average_within_between_ratio(linear_kpca.embedding(), iris, linear_kpca.kept_row_indices());
  const double rbf_ratio = average_within_between_ratio(rbf_kpca.embedding(), iris, rbf_kpca.kept_row_indices());

  // Both should show meaningfully more between-species than within-species separation.
  EXPECT_GT(linear_ratio, 1.2);
  EXPECT_GT(rbf_ratio, 1.2);
}

TEST(KernelPCA, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});

  KernelPCA kpca(frame, {"x", "y"}, KernelPCAOptions{1, "linear", 1.0, 3.0, 1.0});
  EXPECT_EQ(kpca.observations(), 5u);
  EXPECT_EQ(kpca.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(KernelPCA, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(KernelPCA(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(KernelPCA(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(KernelPCA(iris, kIrisFeatures, KernelPCAOptions{0, "rbf", 1.0, 3.0, 1.0}), std::invalid_argument);
  EXPECT_THROW(KernelPCA(iris, kIrisFeatures, KernelPCAOptions{2, "not-a-kernel", 1.0, 3.0, 1.0}),
              std::invalid_argument);
  EXPECT_THROW(KernelPCA(iris, kIrisFeatures, KernelPCAOptions{2, "rbf", -0.5, 3.0, 1.0}), std::invalid_argument);
  EXPECT_THROW(KernelPCA(iris, kIrisFeatures, KernelPCAOptions{2, "rbf", 0.0, 3.0, 1.0}), std::invalid_argument);
  EXPECT_THROW(KernelPCA(iris, kIrisFeatures, KernelPCAOptions{2, "polynomial", -1.0, 3.0, 1.0}),
              std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0});
  EXPECT_THROW(KernelPCA(tiny, {"x"}, KernelPCAOptions{2, "linear", 1.0, 3.0, 1.0}), std::invalid_argument);
}

TEST(KernelPCA, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  KernelPCA kpca(iris, kIrisFeatures, KernelPCAOptions{2, "rbf", 0.13, 3.0, 1.0});

  std::vector<std::string> species(kpca.observations());
  for (std::size_t i = 0; i < kpca.observations(); ++i) species[i] = iris.string_at("Species", kpca.kept_row_indices()[i]);

  const auto grouped = kpca.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, kpca.observations());

  const auto ungrouped = kpca.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(kpca.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(kpca.plot_embedding(10, 0), std::out_of_range);
}
