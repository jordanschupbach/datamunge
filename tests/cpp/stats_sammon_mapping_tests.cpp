#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/mds.hpp>
#include <datamunge/stats/sammon_mapping.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::MDS;
using datamunge::stats::SammonMapping;
using datamunge::stats::SammonMappingOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

double embedded_distance(const datamunge::linalg::DenseMatrix<double>& Y, const std::size_t i, const std::size_t j) {
  double sum = 0.0;
  for (std::size_t k = 0; k < Y.cols(); ++k) {
    const double d = Y(i, k) - Y(j, k);
    sum += d * d;
  }
  return std::sqrt(sum);
}

} // namespace

TEST(SammonMapping, EmbeddingShapeAndFiniteness) {
  const auto iris = datamunge::datasets::iris();
  SammonMapping sm(iris, kIrisFeatures);

  EXPECT_EQ(sm.observations(), iris.nrows());
  EXPECT_EQ(sm.n_components(), 2u);
  EXPECT_EQ(sm.embedding().rows(), iris.nrows());
  EXPECT_EQ(sm.embedding().cols(), 2u);

  for (std::size_t i = 0; i < sm.observations(); ++i)
    for (std::size_t k = 0; k < sm.n_components(); ++k) EXPECT_TRUE(std::isfinite(sm.embedding()(i, k)));

  EXPECT_TRUE(std::isfinite(sm.stress()));
  EXPECT_GT(sm.stress(), 0.0);
}

TEST(SammonMapping, StressConvergesToSmallValue) {
  const auto iris = datamunge::datasets::iris();

  // Stress after essentially no optimization (a single sweep) vs. a full run: the fully-converged
  // stress should be dramatically smaller, demonstrating the pseudo-Newton update actually
  // minimizes the Sammon stress rather than leaving it near its post-initialization value.
  SammonMappingOptions one_iter;
  one_iter.max_iterations = 1;
  SammonMapping sm_one(iris, kIrisFeatures, one_iter);

  SammonMapping sm_full(iris, kIrisFeatures);

  EXPECT_LT(sm_full.stress(), sm_one.stress());
  EXPECT_LT(sm_full.stress(), 0.05) << "Sammon stress on iris should converge to a small value";
  EXPECT_GT(sm_full.iterations_run(), 0u);
}

TEST(SammonMapping, SpeciesAreReasonablySeparated) {
  const auto iris = datamunge::datasets::iris();
  SammonMapping sm(iris, kIrisFeatures);

  std::vector<std::string> species(sm.observations());
  for (std::size_t i = 0; i < sm.observations(); ++i) species[i] = iris.string_at("Species", sm.kept_row_indices()[i]);

  double within_sum = 0.0, between_sum = 0.0;
  std::size_t within_n = 0, between_n = 0;
  for (std::size_t i = 0; i < sm.observations(); ++i) {
    for (std::size_t j = i + 1; j < sm.observations(); ++j) {
      const double d = embedded_distance(sm.embedding(), i, j);
      if (species[i] == species[j]) {
        within_sum += d;
        ++within_n;
      } else {
        between_sum += d;
        ++between_n;
      }
    }
  }
  const double mean_within = within_sum / static_cast<double>(within_n);
  const double mean_between = between_sum / static_cast<double>(between_n);
  EXPECT_LT(mean_within, mean_between) << "Same-species points should be embedded closer together on average";
}

TEST(SammonMapping, RoughlyCorrelatesWithClassicalMds) {
  // Both Sammon mapping and classical MDS are distance-preservation methods on the same pairwise
  // distances, so pairwise embedded distances from each should be strongly (loosely, not exactly)
  // correlated even though the two optimize different loss functions.
  const auto iris = datamunge::datasets::iris();
  SammonMapping sm(iris, kIrisFeatures);
  MDS mds(iris, kIrisFeatures);

  std::vector<double> sammon_d, mds_d;
  for (std::size_t i = 0; i < sm.observations(); ++i) {
    for (std::size_t j = i + 1; j < sm.observations(); ++j) {
      sammon_d.push_back(embedded_distance(sm.embedding(), i, j));
      mds_d.push_back(embedded_distance(mds.embedding(), i, j));
    }
  }

  const double n = static_cast<double>(sammon_d.size());
  double mean_s = 0.0, mean_m = 0.0;
  for (std::size_t i = 0; i < sammon_d.size(); ++i) {
    mean_s += sammon_d[i];
    mean_m += mds_d[i];
  }
  mean_s /= n;
  mean_m /= n;
  double cov = 0.0, var_s = 0.0, var_m = 0.0;
  for (std::size_t i = 0; i < sammon_d.size(); ++i) {
    const double ds = sammon_d[i] - mean_s;
    const double dm = mds_d[i] - mean_m;
    cov += ds * dm;
    var_s += ds * ds;
    var_m += dm * dm;
  }
  const double correlation = cov / std::sqrt(var_s * var_m);
  EXPECT_GT(correlation, 0.8) << "Sammon and MDS embeddings should have roughly correlated pairwise distances";
}

TEST(SammonMapping, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});

  SammonMapping sm(frame, {"x", "y"}, SammonMappingOptions{1, 0.3, 200, 1e-9, DistanceMetric::Euclidean, 42});
  EXPECT_EQ(sm.observations(), 5u);
  EXPECT_EQ(sm.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(SammonMapping, DeterministicGivenSameSeed) {
  const auto iris = datamunge::datasets::iris();
  SammonMapping a(iris, kIrisFeatures);
  SammonMapping b(iris, kIrisFeatures);
  for (std::size_t i = 0; i < a.observations(); ++i)
    for (std::size_t k = 0; k < a.n_components(); ++k) EXPECT_DOUBLE_EQ(a.embedding()(i, k), b.embedding()(i, k));
}

TEST(SammonMapping, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(SammonMapping(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(SammonMapping(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(SammonMapping(iris, kIrisFeatures, SammonMappingOptions{0}), std::invalid_argument);

  SammonMappingOptions bad_lr;
  bad_lr.learning_rate = -0.1;
  EXPECT_THROW(SammonMapping(iris, kIrisFeatures, bad_lr), std::invalid_argument);
  bad_lr.learning_rate = 0.0;
  EXPECT_THROW(SammonMapping(iris, kIrisFeatures, bad_lr), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0});
  EXPECT_THROW(SammonMapping(tiny, {"x"}, SammonMappingOptions{2}), std::invalid_argument);
}

TEST(SammonMapping, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  SammonMapping sm(iris, kIrisFeatures);

  std::vector<std::string> species(sm.observations());
  for (std::size_t i = 0; i < sm.observations(); ++i) species[i] = iris.string_at("Species", sm.kept_row_indices()[i]);

  const auto grouped = sm.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, sm.observations());

  const auto ungrouped = sm.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(sm.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(sm.plot_embedding(10, 0), std::out_of_range);
}
