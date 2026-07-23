#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/pca.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::PCA;
using datamunge::stats::PCAOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

} // namespace

TEST(PCA, MatchesKnownIrisVarianceExplained) {
  const auto iris = datamunge::datasets::iris();
  PCA pca(iris, kIrisFeatures);

  EXPECT_EQ(pca.observations(), iris.nrows());
  EXPECT_EQ(pca.num_components(), 4u);

  const auto ratio = pca.explained_variance_ratio();
  const double total = std::accumulate(ratio.begin(), ratio.end(), 0.0);
  EXPECT_NEAR(total, 1.0, 1e-9);

  // Well-known published values for scaled (correlation-matrix) PCA on the iris dataset.
  EXPECT_NEAR(ratio[0], 0.7296, 1e-3);
  EXPECT_NEAR(ratio[1], 0.2285, 1e-3);

  const auto cumulative = pca.cumulative_explained_variance_ratio();
  EXPECT_NEAR(cumulative.back(), 1.0, 1e-9);
  EXPECT_NEAR(cumulative[1], ratio[0] + ratio[1], 1e-9);
}

TEST(PCA, LoadingsAreOrthonormal) {
  const auto iris = datamunge::datasets::iris();
  PCA pca(iris, kIrisFeatures);

  for (std::size_t c = 0; c < pca.num_components(); ++c) {
    const auto loadings = pca.component_loadings(c);
    double norm_sq = 0.0;
    for (const auto v : loadings) norm_sq += v * v;
    EXPECT_NEAR(norm_sq, 1.0, 1e-8) << "component " << c << " loading vector should be unit length";
  }

  const auto l0 = pca.component_loadings(0);
  const auto l1 = pca.component_loadings(1);
  double dot = 0.0;
  for (std::size_t i = 0; i < l0.size(); ++i) dot += l0[i] * l1[i];
  EXPECT_NEAR(dot, 0.0, 1e-8) << "distinct principal axes should be orthogonal";
}

TEST(PCA, TransformReproducesScoresOnTrainingData) {
  const auto iris = datamunge::datasets::iris();
  PCA pca(iris, kIrisFeatures);

  const auto reprojected = pca.transform(iris);
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t c = 0; c < pca.num_components(); ++c) {
      EXPECT_NEAR(reprojected(i, c), pca.scores()(i, c), 1e-9);
    }
  }
}

TEST(PCA, UnscaledUsesCovarianceNotCorrelation) {
  const auto iris = datamunge::datasets::iris();
  PCA scaled(iris, kIrisFeatures, PCAOptions{true, true});
  PCA unscaled(iris, kIrisFeatures, PCAOptions{true, false});

  // Sepal.Length has a much larger raw variance than the others, so unscaled PCA's first
  // component should explain a larger share of variance than scaled PCA's.
  EXPECT_GT(unscaled.explained_variance_ratio()[0], scaled.explained_variance_ratio()[0]);
}

TEST(PCA, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0});

  PCA pca(frame, {"x", "y"});
  EXPECT_EQ(pca.observations(), 4u);
  EXPECT_EQ(pca.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4}));
}

TEST(PCA, RejectsMissingOrNonNumericColumnsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(PCA(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(PCA(iris, {"Species"}), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0});
  EXPECT_THROW(PCA(tiny, {"x"}), std::invalid_argument);
}

TEST(PCA, PlotScoresGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  PCA pca(iris, kIrisFeatures);

  std::vector<std::string> species(pca.observations());
  for (std::size_t i = 0; i < pca.observations(); ++i) species[i] = iris.string_at("Species", pca.kept_row_indices()[i]);

  const auto grouped = pca.plot_scores(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, pca.observations());

  const auto ungrouped = pca.plot_scores();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  const auto scree = pca.plot_scree();
  EXPECT_EQ(scree.series().size(), 1u);
  EXPECT_EQ(scree.series()[0].y.size(), pca.num_components());

  EXPECT_THROW(pca.plot_scores(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(pca.plot_scores(10, 0), std::out_of_range);
}
