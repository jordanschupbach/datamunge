#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/factor_analysis.hpp>
#include <limits>
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
  std::mt19937_64                  rng(seed);
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
  return DataFrame{
      {"v1", DataFrame::numeric_column_type(cols[0])},
      {"v2", DataFrame::numeric_column_type(cols[1])},
      {"v3", DataFrame::numeric_column_type(cols[2])},
      {"v4", DataFrame::numeric_column_type(cols[3])},
      {"v5", DataFrame::numeric_column_type(cols[4])},
      {"v6", DataFrame::numeric_column_type(cols[5])}};
}

} // namespace

TEST(FactorAnalysis, RecoversKnownBlockStructure) {
  const auto     data = make_two_factor();
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
  for (std::size_t j = 0; j < 6; ++j)
    EXPECT_NEAR(fa.communalities()[j], true_comm[j], 0.08);
}

TEST(FactorAnalysis, CommunalityPlusUniquenessIsOne) {
  const auto     data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});
  for (std::size_t j = 0; j < 6; ++j) {
    // On the correlation scale communality + uniqueness == 1 (uniqueness clamped at 0 in Heywood).
    EXPECT_NEAR(fa.communalities()[j] + fa.uniquenesses()[j], 1.0, 1e-9);
    EXPECT_GE(fa.uniquenesses()[j], 0.0);
    // Communality is the loadings' row sum of squares.
    double ss = 0.0;
    for (std::size_t c = 0; c < fa.num_factors(); ++c)
      ss += fa.loadings()(j, c) * fa.loadings()(j, c);
    EXPECT_NEAR(fa.communalities()[j], ss, 1e-9);
  }
}

TEST(FactorAnalysis, RotationPreservesCommunalitiesAndFit) {
  const auto     data = make_two_factor();
  FactorAnalysis raw(data, kVars, FactorAnalysisOptions{2, true, "none", 1000, 1e-6});
  FactorAnalysis rot(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  // Rotation is orthogonal: per-variable communalities are invariant...
  for (std::size_t j = 0; j < 6; ++j)
    EXPECT_NEAR(raw.communalities()[j], rot.communalities()[j], 1e-6);

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
  const auto     data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  // Sample correlation of the data.
  const std::size_t                p = 6, n = data.nrows();
  std::vector<std::vector<double>> z(p, std::vector<double>(n));
  for (std::size_t j = 0; j < p; ++j) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      m += data.double_at(kVars[j], i);
    m /= n;
    double sd = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      sd += (data.double_at(kVars[j], i) - m) * (data.double_at(kVars[j], i) - m);
    sd = std::sqrt(sd / (n - 1));
    for (std::size_t i = 0; i < n; ++i)
      z[j][i] = (data.double_at(kVars[j], i) - m) / sd;
  }
  double      ss  = 0.0;
  std::size_t cnt = 0;
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t b = 0; b < p; ++b) {
      if (a == b)
        continue;
      double r = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        r += z[a][i] * z[b][i];
      r /= (n - 1);
      double fit = 0.0;
      for (std::size_t c = 0; c < 2; ++c)
        fit += fa.loadings()(a, c) * fa.loadings()(b, c);
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
  const auto                     iris  = datamunge::datasets::iris();
  const std::vector<std::string> feats = {
      "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};
  EXPECT_THROW(FactorAnalysis(iris, {}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(
      FactorAnalysis(iris, feats, FactorAnalysisOptions{0, true, "none", 1000, 1e-6}),
      std::invalid_argument);
  EXPECT_THROW(
      FactorAnalysis(iris, feats, FactorAnalysisOptions{5, true, "none", 1000, 1e-6}),
      std::invalid_argument);
  EXPECT_THROW(
      FactorAnalysis(iris, feats, FactorAnalysisOptions{2, true, "oblimin", 1000, 1e-6}),
      std::invalid_argument);
}

TEST(FactorAnalysis, PlotsAndAccessorsGuardRanges) {
  const auto     data = make_two_factor();
  FactorAnalysis fa(data, kVars, FactorAnalysisOptions{2, true, "varimax", 1000, 1e-6});

  const auto loadings_plot = fa.plot_loadings();
  EXPECT_FALSE(loadings_plot.series().empty());

  std::vector<std::string> labels(fa.observations(), "grp");
  const auto               scores_plot = fa.plot_scores(labels);
  EXPECT_EQ(scores_plot.series().size(), 1u);

  EXPECT_THROW(fa.plot_scores(std::vector<std::string>{"too-few"}), std::invalid_argument);
  EXPECT_THROW(fa.plot_loadings(0, 9), std::out_of_range);
  EXPECT_THROW(fa.factor_loadings(9), std::out_of_range);
  EXPECT_THROW(fa.factor_scores(9), std::out_of_range);
}

namespace {
DataFrame exact_three(double scale = 1) {
  const double a = scale * std::sqrt(7.) / 2;
  DataFrame    out;
  out.add_column("a", std::vector<double>{a, a, 0, 0, 0, 0, -a, -a});
  out.add_column("b", std::vector<double>{a, 0, a, 0, 0, -a, 0, -a});
  out.add_column("c", std::vector<double>{a, 0, 0, a, -a, 0, 0, -a});
  return out;
}
} // namespace
TEST(FactorAnalysis, ExactThreeIndicatorsAndIterationRecurrence) {
  const auto           d = exact_three();
  const FactorAnalysis m(d, {"a", "b", "c"}, {1, true, "none", 1000, 1e-12});
  ASSERT_TRUE(m.converged());
  const auto& h = m.iteration_history();
  ASSERT_EQ(h.nrows(), m.iterations());
  for (std::size_t t = 0; t < h.nrows(); ++t)
    EXPECT_NEAR(h.double_at("communality_1", t), .5 - std::pow(1. / 3, t + 1) / 6, 2e-12);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(m.means()[i], 0, 1e-15);
    EXPECT_NEAR(m.scales()[i], 1, 1e-15);
    EXPECT_NEAR(m.communalities()[i], .5, 2e-12);
    EXPECT_NEAR(m.loadings()(i, 0), std::sqrt(.5), 2e-12);
    EXPECT_NEAR(m.uniquenesses()[i], .5, 2e-12);
  }
  EXPECT_NEAR(m.diagnostics().double_at("degrees_freedom", 0), 0, 0);
  EXPECT_NEAR(m.diagnostics().double_at("offdiagonal_rms", 0), 0, 2e-12);
  EXPECT_EQ(m.loading_table().nrows(), 3);
  EXPECT_EQ(m.variance_table().nrows(), 3);
  EXPECT_EQ(m.covariance_pairs().nrows(), 6);
  // C^-1*Lambda=(sqrt(.5)/2)*ones; scores are shrunken regression estimates.
  for (std::size_t i = 0; i < d.nrows(); ++i)
    EXPECT_NEAR(
        m.scores()(i, 0),
        (d.double_at("a", i) + d.double_at("b", i) + d.double_at("c", i)) * std::sqrt(.5) / 2,
        1e-11);
}
TEST(FactorAnalysis, CovarianceInitializationHasVarianceUnits) {
  const auto           d = exact_three(2);
  const FactorAnalysis first(d, {"a", "b", "c"}, {1, false, "none", 1, 1e-12});
  EXPECT_FALSE(first.converged());
  EXPECT_NEAR(first.iteration_history().double_at("communality_1", 0), 16. / 9, 1e-12);
  EXPECT_NEAR(first.proportion_variance()[0], 4. / 9, 1e-12);
  const FactorAnalysis full(d, {"a", "b", "c"}, {1, false, "none", 1000, 1e-12});
  for (auto h : full.communalities())
    EXPECT_NEAR(h, 2, 2e-12);
  EXPECT_NEAR(full.diagnostics().double_at("total_variance", 0), 12, 1e-12);
  EXPECT_NEAR(full.proportion_variance()[0], .5, 2e-12);
}
TEST(FactorAnalysis, BoundaryEvidenceDoesNotConfuseStoppingWithAdmissibility) {
  const auto                     d     = datamunge::datasets::iris();
  const std::vector<std::string> names = {
      "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};
  const FactorAnalysis one(d, names, {1, true, "none"});
  ASSERT_TRUE(one.converged());
  EXPECT_GE(one.diagnostics().double_at("heywood_variables", 0), 1);
  EXPECT_EQ(one.diagnostics().double_at("degrees_freedom", 0), 2);
  const auto v = one.variance_table();
  EXPECT_LT(v.double_at("raw_uniqueness", 2), 0);
  EXPECT_EQ(one.uniquenesses()[2], 0);
  EXPECT_GT(one.reproduced_covariance()(2, 2), one.covariance_matrix()(2, 2));
  EXPECT_NEAR(
      one.diagnostics().double_at("max_diagonal_excess", 0),
      -v.double_at("raw_uniqueness", 2),
      1e-13);
  const FactorAnalysis two(d, names, {2, true, "none", 2, 1e-12});
  EXPECT_EQ(two.diagnostics().double_at("degrees_freedom", 0), -1);
  EXPECT_FALSE(two.converged());
  EXPECT_EQ(two.iterations(), 2);
}
TEST(FactorAnalysis, RotationCriterionAndCommonCovarianceAreInspectable) {
  const auto           d = make_two_factor();
  const FactorAnalysis raw(d, kVars, {2, true, "none"}), rot(d, kVars, {2, true, "varimax"});
  const auto           a = raw.common_covariance(), b = rot.common_covariance();
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      EXPECT_NEAR(a(i, j), b(i, j), 1e-11);
  EXPECT_GE(
      rot.diagnostics().double_at("kaiser_varimax", 0),
      raw.diagnostics().double_at("kaiser_varimax", 0) - 1e-12);
  EXPECT_EQ(raw.iterations(), rot.iterations());
  EXPECT_THROW((void)rot.plot_loadings(0, 0), std::invalid_argument);
}
TEST(FactorAnalysis, RejectsUndefinedOrNonfiniteFits) {
  const auto d = exact_three();
  EXPECT_THROW(FactorAnalysis(d, {"a", "a"}, {1}), std::invalid_argument);
  EXPECT_THROW(FactorAnalysis(d, {"a"}, {1, true, "none", 0}), std::invalid_argument);
  for (double tol :
       {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(FactorAnalysis(d, {"a"}, {1, true, "none", 10, tol}), std::invalid_argument);
  DataFrame constant;
  constant.add_column("a", std::vector<double>{2, 2, 2});
  EXPECT_THROW(FactorAnalysis(constant, {"a"}, {1}), std::invalid_argument);
  DataFrame invalid;
  invalid.add_column("a", std::vector<double>{1, std::numeric_limits<double>::infinity(), 3});
  EXPECT_THROW(FactorAnalysis(invalid, {"a"}, {1}), std::invalid_argument);
  DataFrame partial;
  partial.add_column("a", std::vector<std::optional<double>>{1., 2., std::nullopt, 4.});
  partial.add_column("b", std::vector<double>{2, 3, std::numeric_limits<double>::infinity(), 7});
  const FactorAnalysis retained(partial, {"a", "b"}, {1});
  EXPECT_EQ(retained.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3}));
}
TEST(FactorAnalysis, SamplesKnownPopulationCovarianceWithLocalSeed) {
  using datamunge::stats::factor_model_sample;
  const datamunge::linalg::DenseMatrix<double> L(3, 2, {.8, .1, .4, .6, -.3, .7});
  const std::vector<double>                    psi = {.3, .2, .4};
  const auto                                   d   = factor_model_sample(L, psi, 20000, 41),
             again                                 = factor_model_sample(L, psi, 20000, 41);
  for (std::size_t j = 0; j < 3; ++j) {
    const auto name = "v" + std::to_string(j + 1);
    EXPECT_EQ(d.double_at(name, 17), again.double_at(name, 17));
  }
  const FactorAnalysis m(d, {"v1", "v2", "v3"}, {1, false, "none"});
  const auto           common = L * L.transpose();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      EXPECT_NEAR(m.covariance_matrix()(i, j), common(i, j) + (i == j ? psi[i] : 0), .03);
  EXPECT_THROW((void)factor_model_sample(L, {.2}, 10), std::invalid_argument);
  EXPECT_THROW((void)factor_model_sample(L, {.2, -1, .3}, 10), std::invalid_argument);
  EXPECT_THROW((void)factor_model_sample(L, psi, 1), std::invalid_argument);
}
TEST(FactorAnalysis, CovarianceResidualRmsChecksShapeAndFiniteValues) {
  using datamunge::linalg::DenseMatrix;
  using datamunge::stats::covariance_residual_rms;
  const DenseMatrix<double> a(2, 2, {1, 2, 3, 4}), b(2, 2, {8, 0, 0, 9});
  EXPECT_NEAR(covariance_residual_rms(a, b), std::sqrt(6.5), 1e-15);
  EXPECT_EQ(covariance_residual_rms(DenseMatrix<double>(1, 1, 4), DenseMatrix<double>(1, 1, 0)), 0);
  EXPECT_THROW((void)covariance_residual_rms(a, DenseMatrix<double>(2, 1)), std::invalid_argument);
  EXPECT_THROW(
      (void)covariance_residual_rms(
          a, DenseMatrix<double>(2, 2, std::numeric_limits<double>::infinity())),
      std::invalid_argument);
}
