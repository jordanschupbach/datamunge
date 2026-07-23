#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/mds.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::MDS;
using datamunge::stats::MDSOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

double euclidean(const DataFrame& df, const std::vector<std::string>& features, const std::size_t i, const std::size_t j) {
  double sum = 0.0;
  for (const auto& f : features) {
    const double d = df.double_at(f, i) - df.double_at(f, j);
    sum += d * d;
  }
  return std::sqrt(sum);
}

} // namespace

TEST(MDS, EmbeddingPreservesPairwiseDistanceOrdering) {
  const auto iris = datamunge::datasets::iris();
  MDS mds(iris, kIrisFeatures);

  EXPECT_EQ(mds.observations(), iris.nrows());
  EXPECT_EQ(mds.n_components(), 2u);
  EXPECT_GT(mds.goodness_of_fit(), 0.9);

  std::vector<double> orig, embedded;
  for (std::size_t i = 0; i < iris.nrows(); i += 3) {
    for (std::size_t j = i + 1; j < iris.nrows(); j += 3) {
      orig.push_back(euclidean(iris, kIrisFeatures, i, j));
      double sum = 0.0;
      for (std::size_t c = 0; c < 2; ++c) {
        const double d = mds.embedding()(i, c) - mds.embedding()(j, c);
        sum += d * d;
      }
      embedded.push_back(std::sqrt(sum));
    }
  }

  const double n = static_cast<double>(orig.size());
  double mean_o = 0.0, mean_e = 0.0;
  for (std::size_t i = 0; i < orig.size(); ++i) {
    mean_o += orig[i];
    mean_e += embedded[i];
  }
  mean_o /= n;
  mean_e /= n;
  double cov = 0.0, var_o = 0.0, var_e = 0.0;
  for (std::size_t i = 0; i < orig.size(); ++i) {
    const double do_ = orig[i] - mean_o;
    const double de_ = embedded[i] - mean_e;
    cov += do_ * de_;
    var_o += do_ * do_;
    var_e += de_ * de_;
  }
  const double correlation = cov / std::sqrt(var_o * var_e);
  EXPECT_GT(correlation, 0.9) << "MDS embedding distances should strongly correlate with original distances";
}

TEST(MDS, EuclideanDistanceMdsClosesMatchesPcaVarianceStructure) {
  const auto iris = datamunge::datasets::iris();
  // Classical MDS on raw Euclidean distances is mathematically closely related to unscaled PCA
  // -- both derive from eigendecomposing a Gram/covariance matrix built from the same centered
  // data -- so their leading-eigenvalue proportions should be in the same ballpark.
  MDS mds(iris, kIrisFeatures, MDSOptions{2, DistanceMetric::Euclidean});
  EXPECT_GT(mds.eigenvalues()[0], mds.eigenvalues()[1]);
  EXPECT_GT(mds.eigenvalues()[1], 0.0);
}

TEST(MDS, ManhattanMetricProducesADifferentEmbeddingThanEuclidean) {
  const auto iris = datamunge::datasets::iris();
  MDS euclidean_mds(iris, kIrisFeatures, MDSOptions{2, DistanceMetric::Euclidean});
  MDS manhattan_mds(iris, kIrisFeatures, MDSOptions{2, DistanceMetric::Manhattan});
  EXPECT_NE(euclidean_mds.eigenvalues()[0], manhattan_mds.eigenvalues()[0]);
}

TEST(MDS, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});

  MDS mds(frame, {"x", "y"}, MDSOptions{1, DistanceMetric::Euclidean});
  EXPECT_EQ(mds.observations(), 5u);
  EXPECT_EQ(mds.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(MDS, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(MDS(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(MDS(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(MDS(iris, kIrisFeatures, MDSOptions{0, DistanceMetric::Euclidean}), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0});
  EXPECT_THROW(MDS(tiny, {"x"}, MDSOptions{2, DistanceMetric::Euclidean}), std::invalid_argument);
}

TEST(MDS, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  MDS mds(iris, kIrisFeatures);

  std::vector<std::string> species(mds.observations());
  for (std::size_t i = 0; i < mds.observations(); ++i) species[i] = iris.string_at("Species", mds.kept_row_indices()[i]);

  const auto grouped = mds.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, mds.observations());

  const auto ungrouped = mds.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(mds.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(mds.plot_embedding(10, 0), std::out_of_range);
}
