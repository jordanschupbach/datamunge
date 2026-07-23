#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/isomap.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::Isomap;
using datamunge::stats::IsomapOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

// The full 150-row / 4-feature iris dataset's symmetric k-NN graph is only fully connected from
// n_neighbors=25 up (verified empirically: setosa is far enough from versicolor/virginica that
// smaller k values -- including the 10-15 range that would be "reasonable" for many datasets --
// leave the graph split into a 50-point (setosa) and a 100-point (versicolor+virginica)
// component). 25 is the smallest value that connects the whole dataset.
constexpr std::size_t kConnectedNeighbors = 25;

} // namespace

TEST(Isomap, EmbeddingHasCorrectShapeAndIsFinite) {
  const auto iris = datamunge::datasets::iris();
  Isomap iso(iris, kIrisFeatures, IsomapOptions{2, kConnectedNeighbors, DistanceMetric::Euclidean});

  EXPECT_EQ(iso.observations(), iris.nrows());
  EXPECT_EQ(iso.n_components(), 2u);
  EXPECT_GT(iso.goodness_of_fit(), 0.9);

  for (std::size_t i = 0; i < iso.observations(); ++i)
    for (std::size_t c = 0; c < iso.n_components(); ++c) EXPECT_TRUE(std::isfinite(iso.embedding()(i, c)));

  ASSERT_EQ(iso.eigenvalues().size(), iso.observations());
  for (std::size_t i = 1; i < iso.eigenvalues().size(); ++i) EXPECT_GE(iso.eigenvalues()[i - 1], iso.eigenvalues()[i]);
}

TEST(Isomap, SpeciesAreReasonablySeparatedInTheEmbedding) {
  const auto iris = datamunge::datasets::iris();
  Isomap iso(iris, kIrisFeatures, IsomapOptions{2, kConnectedNeighbors, DistanceMetric::Euclidean});

  std::vector<std::string> species(iso.observations());
  for (std::size_t i = 0; i < iso.observations(); ++i) species[i] = iris.string_at("Species", iso.kept_row_indices()[i]);

  double within_sum = 0.0, between_sum = 0.0;
  std::size_t within_n = 0, between_n = 0;
  for (std::size_t i = 0; i < iso.observations(); ++i) {
    for (std::size_t j = i + 1; j < iso.observations(); ++j) {
      double sum = 0.0;
      for (std::size_t c = 0; c < iso.n_components(); ++c) {
        const double d = iso.embedding()(i, c) - iso.embedding()(j, c);
        sum += d * d;
      }
      const double dist = std::sqrt(sum);
      if (species[i] == species[j]) {
        within_sum += dist;
        ++within_n;
      } else {
        between_sum += dist;
        ++between_n;
      }
    }
  }
  ASSERT_GT(within_n, 0u);
  ASSERT_GT(between_n, 0u);
  const double within_avg = within_sum / static_cast<double>(within_n);
  const double between_avg = between_sum / static_cast<double>(between_n);
  EXPECT_LT(within_avg, between_avg)
      << "average within-species embedded distance should be smaller than average between-species distance";
}

TEST(Isomap, ThrowsWhenNeighborGraphIsDisconnected) {
  const auto iris = datamunge::datasets::iris();
  // A small n_neighbors leaves the setosa cluster disconnected from the rest of the dataset (see
  // kConnectedNeighbors above) -- Isomap must fail loudly rather than silently embed a broken
  // geodesic-distance matrix.
  EXPECT_THROW(Isomap(iris, kIrisFeatures, IsomapOptions{2, 2, DistanceMetric::Euclidean}), std::invalid_argument);
  EXPECT_THROW(Isomap(iris, kIrisFeatures, IsomapOptions{2, 10, DistanceMetric::Euclidean}), std::invalid_argument);
}

TEST(Isomap, RejectsTinyNNeighborsAtValidationBeforeFitting) {
  const auto iris = datamunge::datasets::iris();
  // n_neighbors must be at least 2 per the interface contract -- this is rejected by input
  // validation (a clear, predictable std::invalid_argument), not the disconnection check.
  EXPECT_THROW(Isomap(iris, kIrisFeatures, IsomapOptions{2, 1, DistanceMetric::Euclidean}), std::invalid_argument);
  EXPECT_THROW(Isomap(iris, kIrisFeatures, IsomapOptions{2, 0, DistanceMetric::Euclidean}), std::invalid_argument);
}

TEST(Isomap, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, 3.0, std::nullopt, 5.0, 6.0, 7.0, 8.0});

  Isomap iso(frame, {"x"}, IsomapOptions{1, 2, DistanceMetric::Euclidean});
  EXPECT_EQ(iso.observations(), 7u);
  EXPECT_EQ(iso.kept_row_indices(), (std::vector<std::size_t>{0, 1, 2, 4, 5, 6, 7}));
  for (std::size_t i = 0; i < iso.observations(); ++i) EXPECT_TRUE(std::isfinite(iso.embedding()(i, 0)));
}

TEST(Isomap, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(Isomap(iris, {"NotAColumn"}, IsomapOptions{2, 5, DistanceMetric::Euclidean}), std::invalid_argument);
  EXPECT_THROW(Isomap(iris, {"Species"}, IsomapOptions{2, 5, DistanceMetric::Euclidean}), std::invalid_argument);
  EXPECT_THROW(Isomap(iris, kIrisFeatures, IsomapOptions{0, 5, DistanceMetric::Euclidean}), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0, 3.0});
  // n_neighbors=5 is more than the 2 other points available.
  EXPECT_THROW(Isomap(tiny, {"x"}, IsomapOptions{1, 5, DistanceMetric::Euclidean}), std::invalid_argument);
}

TEST(Isomap, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  Isomap iso(iris, kIrisFeatures, IsomapOptions{2, kConnectedNeighbors, DistanceMetric::Euclidean});

  std::vector<std::string> species(iso.observations());
  for (std::size_t i = 0; i < iso.observations(); ++i) species[i] = iris.string_at("Species", iso.kept_row_indices()[i]);

  const auto grouped = iso.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, iso.observations());

  const auto ungrouped = iso.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(iso.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(iso.plot_embedding(10, 0), std::out_of_range);
}
