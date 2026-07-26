#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/factor_analysis.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::FactorAnalysis;
using datamunge::stats::FactorAnalysisOptions;

namespace {

const std::vector<std::string> kVars = {"v1", "v2", "v3", "v4", "v5", "v6"};

// Synthetic data from a known two-factor model: variables 1-3 load on factor A, 4-6 on factor B,
// factors independent. Deterministic (fixed seed) so tests are reproducible.
DataFrame make_two_factor(std::size_t n = 500, unsigned seed = 123) {
  const double true_load[6][2] = {
      {0.85, 0.00}, {0.80, 0.00}, {0.75, 0.00}, {0.00, 0.70}, {0.00, 0.65}, {0.00, 0.60}};
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<std::vector<double>> cols(6);
  for (std::size_t i = 0; i < n; ++i) {
    const double fa = z(rng), fb = z(rng);
    for (std::size_t j = 0; j < 6; ++j) {
      const double comm = true_load[j][0] * true_load[j][0] + true_load[j][1] * true_load[j][1];
      const double uniq = std::sqrt(std::max(1.0 - comm, 0.0));
      cols[j].push_back(true_load[j][0] * fa + true_load[j][1] * fb + uniq * z(rng));
    }
  }
  return DataFrame{{"v1", DataFrame::numeric_column_type(cols[0])},
                   {"v2", DataFrame::numeric_column_type(cols[1])},
                   {"v3", DataFrame::numeric_column_type(cols[2])},
                   {"v4", DataFrame::numeric_column_type(cols[3])},
                   {"v5", DataFrame::numeric_column_type(cols[4])},
                   {"v6", DataFrame::numeric_column_type(cols[5])}};
}

} // namespace

TEST(FactorAnalysis, RecoversKnownBlockStructure) {
  const auto data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  EXPECT_EQ(fa.observations(), 500u);
  EXPECT_EQ(fa.num_factors(), 2u);
  EXPECT_TRUE(fa.converged());

  // Each of the first three variables should load most strongly on one factor, and each of the last
  // three on the other -- i.e. the two blocks separate cleanly onto different factors.
  auto dominant = [&](std::size_t j) {
    return std::abs(fa.loadings()(j, 0)) >= std::abs(fa.loadings()(j, 1)) ? 0 : 1;
  };
  const int block1 = dominant(0);
  EXPECT_EQ(dominant(1), block1);
  EXPECT_EQ(dominant(2), block1);
  const int block2 = dominant(3);
  EXPECT_NE(block2, block1);
  EXPECT_EQ(dominant(4), block2);
  EXPECT_EQ(dominant(5), block2);

  // Communalities should track the true ones (0.72, 0.64, 0.56, 0.49, 0.42, 0.36).
  const double true_comm[6] = {0.7225, 0.64, 0.5625, 0.49, 0.4225, 0.36};
  for (std::size_t j = 0; j < 6; ++j) EXPECT_NEAR(fa.communalities()[j], true_comm[j], 0.08);
}

TEST(FactorAnalysis, CommunalityPlusUniquenessIsOne) {
  const auto data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});
  for (std::size_t j = 0; j < 6; ++j) {
    // On the correlation scale communality + uniqueness == 1 (uniqueness clamped at 0 in Heywood).
    EXPECT_NEAR(fa.communalities()[j] + fa.uniquenesses()[j], 1.0, 1e-9);
    EXPECT_GE(fa.uniquenesses()[j], 0.0);
    // Communality is the loadings' row sum of squares.
    double ss = 0.0;
    for (std::size_t c = 0; c < fa.num_factors(); ++c) ss += fa.loadings()(j, c) * fa.loadings()(j, c);
    EXPECT_NEAR(fa.communalities()[j], ss, 1e-9);
  }
}

TEST(FactorAnalysis, RotationPreservesCommunalitiesAndFit) {
  const auto data = make_two_factor();
  FactorAnalysis raw(data, kVars, FactorAnalysisOptions{2, true, "none", 1000, 1e-6});
  FactorAnalysis rot(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  // Rotation is orthogonal: per-variable communalities are invariant...
  for (std::size_t j = 0; j < 6; ++j) EXPECT_NEAR(raw.communalities()[j], rot.communalities()[j], 1e-6);

  // ...and so is the reproduced correlation Lambda Lambda^T.
  for (std::size_t a = 0; a < 6; ++a)
    for (std::size_t b = 0; b < 6; ++b) {
      double sr = 0.0, sv = 0.0;
      for (std::size_t c = 0; c < 2; ++c) {
        sr += raw.loadings()(a, c) * raw.loadings()(b, c);
        sv += rot.loadings()(a, c) * rot.loadings()(b, c);
      }
      EXPECT_NEAR(sr, sv, 1e-6);
    }
}

TEST(FactorAnalysis, ReproducesOffDiagonalCorrelations) {
  const auto data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  // Sample correlation of the data.
  const std::size_t p = 6, n = data.nrows();
  std::vector<std::vector<double>> z(p, std::vector<double>(n));
  for (std::size_t j = 0; j < p; ++j) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m += data.double_at(kVars[j], i);
    m /= n;
    double sd = 0.0;
    for (std::size_t i = 0; i < n; ++i) sd += (data.double_at(kVars[j], i) - m) * (data.double_at(kVars[j], i) - m);
    sd = std::sqrt(sd / (n - 1));
    for (std::size_t i = 0; i < n; ++i) z[j][i] = (data.double_at(kVars[j], i) - m) / sd;
  }
  double ss = 0.0;
  std::size_t cnt = 0;
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t b = 0; b < p; ++b) {
      if (a == b) continue;
      double r = 0.0;
      for (std::size_t i = 0; i < n; ++i) r += z[a][i] * z[b][i];
      r /= (n - 1);
      double fit = 0.0;
      for (std::size_t c = 0; c < 2; ++c) fit += fa.loadings()(a, c) * fa.loadings()(b, c);
      ss += (r - fit) * (r - fit);
      ++cnt;
    }
  EXPECT_LT(std::sqrt(ss / cnt), 0.05);
}

TEST(FactorAnalysis, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});
  FactorAnalysis fa(frame, {"x", "y"}, FactorAnalysisOptions{1, true, "none", 1000, 1e-6});
  EXPECT_EQ(fa.observations(), 5u);
  EXPECT_EQ(fa.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(FactorAnalysis, RejectsInvalidOptions) {
  const auto iris = datamunge::datasets::iris();
  const std::vector<std::string> feats = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};
  EXPECT_THROW(FactorAnalysis(iris, {}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, feats, FactorAnalysisOptions{0, true, "none", 1000, 1e-6}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, feats, FactorAnalysisOptions{5, true, "none", 1000, 1e-6}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, feats, FactorAnalysisOptions{2, true, "oblimin", 1000, 1e-6}),
               std::invalid_argument);
}

TEST(FactorAnalysis, PlotsAndAccessorsGuardRanges) {
  const auto data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  const auto loadings_plot = fa.plot_loadings();
  EXPECT_FALSE(loadings_plot.series().empty());

  std::vector<std::string> labels(fa.observations(), "grp");
  const auto scores_plot = fa.plot_scores(labels);
  EXPECT_EQ(scores_plot.series().size(), 1u);

  EXPECT_THROW(fa.plot_scores(std::vector<std::string>{"too-few"}), std::invalid_argument);
  EXPECT_THROW(fa.plot_loadings(0, 9), std::out_of_range);
  EXPECT_THROW(fa.factor_loadings(9), std::out_of_range);
  EXPECT_THROW(fa.factor_scores(9), std::out_of_range);
}
