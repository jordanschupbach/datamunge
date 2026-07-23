#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/diffusion_maps.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DiffusionMaps;
using datamunge::stats::DiffusionMapsOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

// Empirically, the median squared pairwise Euclidean distance across the iris feature columns is
// ~5.57 (mean ~9.15); a heat kernel bandwidth on that order gives a well-behaved kernel (neither
// collapsed to the identity nor washed out to uniform).
constexpr double kIrisEpsilon = 5.57;

double within_between_ratio(const DiffusionMaps& model, const std::vector<std::string>& species) {
  double within_sum = 0.0, within_n = 0.0, between_sum = 0.0, between_n = 0.0;
  for (std::size_t i = 0; i < model.observations(); ++i) {
    for (std::size_t j = i + 1; j < model.observations(); ++j) {
      double sum = 0.0;
      for (std::size_t c = 0; c < model.n_components(); ++c) {
        const double d = model.embedding()(i, c) - model.embedding()(j, c);
        sum += d * d;
      }
      const double dist = std::sqrt(sum);
      if (species[i] == species[j]) {
        within_sum += dist;
        within_n += 1;
      } else {
        between_sum += dist;
        between_n += 1;
      }
    }
  }
  return (between_sum / between_n) / (within_sum / within_n);
}

} // namespace

TEST(DiffusionMaps, EmbeddingHasExpectedShapeAndIsFinite) {
  const auto iris = datamunge::datasets::iris();
  DiffusionMaps dm(iris, kIrisFeatures, DiffusionMapsOptions{2, kIrisEpsilon, 0.5, 1.0});

  EXPECT_EQ(dm.observations(), iris.nrows());
  EXPECT_EQ(dm.n_components(), 2u);
  EXPECT_EQ(dm.embedding().rows(), iris.nrows());
  EXPECT_EQ(dm.embedding().cols(), 2u);

  for (std::size_t i = 0; i < dm.observations(); ++i)
    for (std::size_t c = 0; c < dm.n_components(); ++c) EXPECT_TRUE(std::isfinite(dm.embedding()(i, c)));
}

TEST(DiffusionMaps, TopEigenvalueOfSymmetrizedMarkovMatrixIsApproximatelyOne) {
  const auto iris = datamunge::datasets::iris();
  DiffusionMaps dm(iris, kIrisFeatures, DiffusionMapsOptions{2, kIrisEpsilon, 0.5, 1.0});

  ASSERT_FALSE(dm.eigenvalues().empty());
  EXPECT_NEAR(dm.eigenvalues()[0], 1.0, 1e-9);
  // Eigenvalues are descending, so everything after the trivial top one should be <= it.
  for (std::size_t i = 1; i < dm.eigenvalues().size(); ++i) EXPECT_LE(dm.eigenvalues()[i], dm.eigenvalues()[i - 1] + 1e-9);
}

TEST(DiffusionMaps, SpeciesAreReasonablySeparatedInTheEmbedding) {
  const auto iris = datamunge::datasets::iris();
  DiffusionMaps dm(iris, kIrisFeatures, DiffusionMapsOptions{2, kIrisEpsilon, 0.5, 1.0});

  std::vector<std::string> species(dm.observations());
  for (std::size_t i = 0; i < dm.observations(); ++i) species[i] = iris.string_at("Species", dm.kept_row_indices()[i]);

  // Average between-species embedded distance should clearly exceed average within-species
  // embedded distance if the manifold structure separating the 3 iris species survived the
  // embedding.
  EXPECT_GT(within_between_ratio(dm, species), 1.5);
}

TEST(DiffusionMaps, AlphaZeroAndAlphaOneProduceGenuinelyDifferentEmbeddings) {
  const auto iris = datamunge::datasets::iris();
  DiffusionMaps dm_alpha0(iris, kIrisFeatures, DiffusionMapsOptions{2, kIrisEpsilon, 0.0, 1.0});
  DiffusionMaps dm_alpha1(iris, kIrisFeatures, DiffusionMapsOptions{2, kIrisEpsilon, 1.0, 1.0});

  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < dm_alpha0.observations(); ++i)
    for (std::size_t c = 0; c < dm_alpha0.n_components(); ++c)
      max_abs_diff = std::max(max_abs_diff, std::abs(dm_alpha0.embedding()(i, c) - dm_alpha1.embedding()(i, c)));

  // The alpha-normalization step is the key Coifman-Lafon innovation -- if it were a no-op, these
  // two embeddings would be identical (or merely a rescaling coincidence). Empirically the two
  // embeddings differ substantially (max abs coordinate difference ~0.72 on iris).
  EXPECT_GT(max_abs_diff, 0.01);
  EXPECT_NE(dm_alpha0.eigenvalues()[1], dm_alpha1.eigenvalues()[1]);
}

TEST(DiffusionMaps, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});

  DiffusionMaps dm(frame, {"x", "y"}, DiffusionMapsOptions{1, 1.0, 0.5, 1.0});
  EXPECT_EQ(dm.observations(), 5u);
  EXPECT_EQ(dm.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(DiffusionMaps, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(DiffusionMaps(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(DiffusionMaps(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(DiffusionMaps(iris, kIrisFeatures, DiffusionMapsOptions{0, 1.0, 0.5, 1.0}), std::invalid_argument);
  EXPECT_THROW(DiffusionMaps(iris, kIrisFeatures, DiffusionMapsOptions{2, 0.0, 0.5, 1.0}), std::invalid_argument);
  EXPECT_THROW(DiffusionMaps(iris, kIrisFeatures, DiffusionMapsOptions{2, -1.0, 0.5, 1.0}), std::invalid_argument);
  EXPECT_THROW(DiffusionMaps(iris, kIrisFeatures, DiffusionMapsOptions{2, 1.0, -0.1, 1.0}), std::invalid_argument);
  EXPECT_THROW(DiffusionMaps(iris, kIrisFeatures, DiffusionMapsOptions{2, 1.0, 1.1, 1.0}), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0});
  EXPECT_THROW(DiffusionMaps(tiny, {"x"}, DiffusionMapsOptions{1, 1.0, 0.5, 1.0}), std::invalid_argument);
}

TEST(DiffusionMaps, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  DiffusionMaps dm(iris, kIrisFeatures, DiffusionMapsOptions{2, kIrisEpsilon, 0.5, 1.0});

  std::vector<std::string> species(dm.observations());
  for (std::size_t i = 0; i < dm.observations(); ++i) species[i] = iris.string_at("Species", dm.kept_row_indices()[i]);

  const auto grouped = dm.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, dm.observations());

  const auto ungrouped = dm.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(dm.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(dm.plot_embedding(10, 0), std::out_of_range);
}
