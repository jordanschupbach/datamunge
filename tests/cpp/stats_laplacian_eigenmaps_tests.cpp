#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/laplacian_eigenmaps.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LaplacianEigenmaps;
using datamunge::stats::LaplacianEigenmapsOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

} // namespace

TEST(LaplacianEigenmaps, EmbeddingHasExpectedShapeAndIsFinite) {
  const auto iris = datamunge::datasets::iris();
  LaplacianEigenmaps le(iris, kIrisFeatures);

  EXPECT_EQ(le.observations(), iris.nrows());
  EXPECT_EQ(le.n_components(), 2u);
  EXPECT_EQ(le.eigenvalues().size(), 2u);

  for (std::size_t i = 0; i < le.observations(); ++i)
    for (std::size_t c = 0; c < le.n_components(); ++c) EXPECT_TRUE(std::isfinite(le.embedding()(i, c)));

  // The trivial (discarded) eigenvalue of the symmetric normalized Laplacian is 0; every
  // eigenvalue we actually keep should be non-negative (L_sym is positive semi-definite) and,
  // ascending, no smaller than the eigenvalue that comes before it.
  for (std::size_t c = 0; c < le.eigenvalues().size(); ++c) EXPECT_GE(le.eigenvalues()[c], -1e-9);
  for (std::size_t c = 1; c < le.eigenvalues().size(); ++c) EXPECT_GE(le.eigenvalues()[c], le.eigenvalues()[c - 1] - 1e-9);
}

TEST(LaplacianEigenmaps, SeparatesIrisSpeciesInEmbeddingSpace) {
  const auto iris = datamunge::datasets::iris();
  // With the default n_neighbors (10), the symmetric k-NN graph on iris is empirically
  // disconnected into two components (setosa vs. versicolor+virginica -- setosa is linearly
  // separable from the other two species), which gives the graph Laplacian a second near-zero
  // eigenvalue on top of the discarded trivial one and swamps the embedding's first dimension
  // with a degenerate "which side of the cut" signal. Using a larger n_neighbors reconnects the
  // graph (verified empirically: components go from 2 to 1 somewhere between n_neighbors=20 and
  // 25 on this dataset) so both embedding dimensions carry real structure.
  LaplacianEigenmapsOptions opts;
  opts.n_components = 2;
  opts.n_neighbors = 25;
  opts.heat_kernel_t = 2.0; // on the order of iris's typical pairwise squared distance (median ~5.6)
  LaplacianEigenmaps le(iris, kIrisFeatures, opts);

  std::vector<std::string> species(le.observations());
  for (std::size_t i = 0; i < le.observations(); ++i) species[i] = iris.string_at("Species", le.kept_row_indices()[i]);

  double within_sum = 0.0, within_count = 0.0;
  double between_sum = 0.0, between_count = 0.0;
  for (std::size_t i = 0; i < le.observations(); ++i) {
    for (std::size_t j = i + 1; j < le.observations(); ++j) {
      double sum = 0.0;
      for (std::size_t c = 0; c < le.n_components(); ++c) {
        const double d = le.embedding()(i, c) - le.embedding()(j, c);
        sum += d * d;
      }
      const double dist = std::sqrt(sum);
      if (species[i] == species[j]) {
        within_sum += dist;
        within_count += 1.0;
      } else {
        between_sum += dist;
        between_count += 1.0;
      }
    }
  }

  const double avg_within = within_sum / within_count;
  const double avg_between = between_sum / between_count;
  EXPECT_LT(avg_within, avg_between) << "average within-species embedded distance should be smaller than "
                                        "average between-species embedded distance";
}

TEST(LaplacianEigenmaps, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});

  LaplacianEigenmaps le(frame, {"x", "y"}, LaplacianEigenmapsOptions{1, 2, 1.0});
  EXPECT_EQ(le.observations(), 5u);
  EXPECT_EQ(le.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(LaplacianEigenmaps, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(LaplacianEigenmaps(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(LaplacianEigenmaps(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(LaplacianEigenmaps(iris, kIrisFeatures, LaplacianEigenmapsOptions{0, 10, 1.0}), std::invalid_argument);
  EXPECT_THROW(LaplacianEigenmaps(iris, kIrisFeatures, LaplacianEigenmapsOptions{2, 1, 1.0}), std::invalid_argument);
  EXPECT_THROW(LaplacianEigenmaps(iris, kIrisFeatures, LaplacianEigenmapsOptions{2, 10, 0.0}), std::invalid_argument);
  EXPECT_THROW(LaplacianEigenmaps(iris, kIrisFeatures, LaplacianEigenmapsOptions{2, 10, -1.0}), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0, 3.0});
  EXPECT_THROW(LaplacianEigenmaps(tiny, {"x"}, LaplacianEigenmapsOptions{1, 10, 1.0}), std::invalid_argument);

  DataFrame four_rows;
  four_rows.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
  four_rows.add_column("y", std::vector<double>{2.0, 1.0, 5.0, 3.0});
  EXPECT_THROW(LaplacianEigenmaps(four_rows, {"x", "y"}, LaplacianEigenmapsOptions{4, 2, 1.0}), std::invalid_argument);
}

TEST(LaplacianEigenmaps, ThrowsOnIsolatedPointFromUnderflowingHeatKernelWeights) {
  const auto iris = datamunge::datasets::iris();
  // heat_kernel_t this tiny makes exp(-D2/heat_kernel_t) underflow to exactly 0.0 for every edge
  // of every point (since the smallest nonzero pairwise D2 in iris is still >> 1e-8), so every
  // point ends up with degree 0 -- the isolated-point degenerate case.
  EXPECT_THROW(LaplacianEigenmaps(iris, kIrisFeatures, LaplacianEigenmapsOptions{2, 10, 1e-8}), std::invalid_argument);
}

TEST(LaplacianEigenmaps, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  LaplacianEigenmaps le(iris, kIrisFeatures);

  std::vector<std::string> species(le.observations());
  for (std::size_t i = 0; i < le.observations(); ++i) species[i] = iris.string_at("Species", le.kept_row_indices()[i]);

  const auto grouped = le.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, le.observations());

  const auto ungrouped = le.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(le.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(le.plot_embedding(10, 0), std::out_of_range);
}
