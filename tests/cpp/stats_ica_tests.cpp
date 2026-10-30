#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/ica.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::ICA;
using datamunge::stats::ICAOptions;

namespace {

const std::vector<std::string> kMix = {"x1", "x2", "x3"};

double abscorr(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n  = a.size();
  double            ma = 0.0, mb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  double num = 0.0, da = 0.0, db = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xa = a[i] - ma, xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  return std::abs(num / std::sqrt(da * db));
}

// Three deterministic non-Gaussian sources (sine, square, sawtooth) linearly mixed into three
// observed signals; the true sources are returned alongside for recovery scoring.
DataFrame make_mixtures(std::vector<std::vector<double>>& sources_true, std::size_t n = 1000) {
  std::vector<double> s1(n), s2(n), s3(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i);
    s1[i]          = std::sin(0.060 * t);
    s2[i]          = (std::sin(0.093 * t) >= 0.0) ? 1.0 : -1.0;
    s3[i]          = std::fmod(0.050 * t, 2.0) - 1.0;
  }
  sources_true                = {s1, s2, s3};
  const double        A[3][3] = {{0.80, 0.30, 0.20}, {0.40, 0.90, 0.50}, {0.60, 0.20, 0.70}};
  std::vector<double> x1(n), x2(n), x3(n);
  for (std::size_t i = 0; i < n; ++i) {
    x1[i] = A[0][0] * s1[i] + A[0][1] * s2[i] + A[0][2] * s3[i];
    x2[i] = A[1][0] * s1[i] + A[1][1] * s2[i] + A[1][2] * s3[i];
    x3[i] = A[2][0] * s1[i] + A[2][1] * s2[i] + A[2][2] * s3[i];
  }
  return DataFrame{
      {"x1", DataFrame::numeric_column_type(x1)},
      {"x2", DataFrame::numeric_column_type(x2)},
      {"x3", DataFrame::numeric_column_type(x3)}};
}

double mean_best_recovery(const std::vector<std::vector<double>>& truth, const ICA& ica) {
  double acc = 0.0;
  for (const auto& s : truth) {
    double best = 0.0;
    for (std::size_t c = 0; c < ica.n_components(); ++c)
      best = std::max(best, abscorr(s, ica.component(c)));
    acc += best;
  }
  return acc / static_cast<double>(truth.size());
}

} // namespace

TEST(ICA, RecoversMixedSources) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  ICA                              ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  EXPECT_EQ(ica.observations(), 1000u);
  EXPECT_EQ(ica.n_components(), 3u);
  EXPECT_TRUE(ica.converged());
  EXPECT_GT(mean_best_recovery(truth, ica), 0.95);
}

TEST(ICA, AllContrastsRecover) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  for (const std::string& g : {"logcosh", "exp", "cube"}) {
    ICA ica(mixed, kMix, ICAOptions{3, g, 300, 1e-6, 42});
    EXPECT_GT(mean_best_recovery(truth, ica), 0.90) << "contrast " << g;
  }
}

TEST(ICA, SourcesAreStandardizedAndDecorrelated) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  ICA                              ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  const std::size_t                n = ica.observations();
  for (std::size_t c = 0; c < 3; ++c) {
    double m = 0.0, ss = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      m += ica.sources()(i, c);
    m /= n;
    for (std::size_t i = 0; i < n; ++i)
      ss += (ica.sources()(i, c) - m) * (ica.sources()(i, c) - m);
    EXPECT_NEAR(m, 0.0, 1e-6);
    EXPECT_NEAR(ss / n, 1.0, 1e-10); // unit variance by whitening
  }
  // Distinct recovered components are (at least) decorrelated.
  EXPECT_LT(abscorr(ica.component(0), ica.component(1)), 1e-3);
  EXPECT_LT(abscorr(ica.component(0), ica.component(2)), 1e-3);
  EXPECT_LT(abscorr(ica.component(1), ica.component(2)), 1e-3);
}

TEST(ICA, TransformReproducesSourcesOnTrainingData) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  ICA                              ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  const auto                       t = ica.transform(mixed);
  ASSERT_EQ(t.rows(), ica.observations());
  ASSERT_EQ(t.cols(), 3u);
  for (std::size_t i = 0; i < ica.observations(); ++i)
    for (std::size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(t(i, c), ica.sources()(i, c), 1e-9);
}

TEST(ICA, MixingMatrixReconstructsTheMixtures) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  ICA                              ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  // Centered mixtures should equal sources * mixing^T (full-rank, 3 components from 3 features).
  double      ss  = 0.0;
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < ica.observations(); ++i)
    for (std::size_t l = 0; l < 3; ++l) {
      double recon = 0.0;
      for (std::size_t c = 0; c < 3; ++c)
        recon += ica.sources()(i, c) * ica.mixing_matrix()(l, c);
      const double centered = mixed.double_at(kMix[l], i) - ica.mean()[l];
      ss += (centered - recon) * (centered - recon);
      ++cnt;
    }
  EXPECT_LT(std::sqrt(ss / cnt), 1e-6);
}

TEST(ICA, DeterministicGivenSeed) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  ICA                              a(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  ICA                              b(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  for (std::size_t i = 0; i < a.observations(); ++i)
    for (std::size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(a.sources()(i, c), b.sources()(i, c), 1e-12);
}

TEST(ICA, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});
  ICA ica(frame, {"x", "y"}, ICAOptions{1, "logcosh", 200, 1e-6, 42});
  EXPECT_EQ(ica.observations(), 5u);
  EXPECT_EQ(ica.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(ICA, RejectsInvalidOptions) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth, 200);
  EXPECT_THROW(ICA(mixed, {}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, kMix, ICAOptions{0, "logcosh", 200, 1e-6, 42}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, kMix, ICAOptions{4, "logcosh", 200, 1e-6, 42}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, kMix, ICAOptions{3, "bogus", 200, 1e-6, 42}), std::invalid_argument);
}

TEST(ICA, PlotAndAccessorsGuardRanges) {
  std::vector<std::vector<double>> truth;
  const auto                       mixed = make_mixtures(truth);
  ICA                              ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  std::vector<std::string>         labels(ica.observations(), "grp");
  EXPECT_EQ(ica.plot_sources(labels).series().size(), 1u);
  EXPECT_THROW(ica.plot_sources(std::vector<std::string>{"too-few"}), std::invalid_argument);
  EXPECT_THROW(ica.plot_sources(labels, 0, 9), std::out_of_range);
  EXPECT_THROW(ica.component(9), std::out_of_range);
}

TEST(ICA, ExactCubeUpdateOnIndependentFourCorners) {
  const datamunge::linalg::DenseMatrix<double> z(4, 2, {-1, -1, -1, 1, 1, -1, 1, 1});
  const auto step = datamunge::stats::ica_fixed_point_step(z, {.8, .6}, "cube");
  EXPECT_NEAR(step.raw_update[0], -1.024, 1e-14);
  EXPECT_NEAR(step.raw_update[1], -.432, 1e-14);
  EXPECT_NEAR(step.direction[1] / step.direction[0], .421875, 1e-14);
  EXPECT_NEAR(step.update_norm, std::hypot(1.024, .432), 1e-14);
  EXPECT_NEAR(
      step.alignment_error, 1 - std::abs(.8 * step.direction[0] + .6 * step.direction[1]), 1e-14);
}
TEST(ICA, PopulationWhiteningAndInverseAreExactOnSmallMixtures) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-1.6, -.4, .4, 1.6});
  d.add_column("y", std::vector<double>{-1.5, .9, -.9, 1.5});
  const ICA m(d, {"x", "y"}, {2, "cube", 200, 1e-10, 7});
  for (std::size_t a = 0; a < 2; ++a)
    for (std::size_t b = 0; b < 2; ++b) {
      double v = 0;
      for (std::size_t i = 0; i < 4; ++i)
        v += m.sources()(i, a) * m.sources()(i, b) / 4;
      EXPECT_NEAR(v, a == b ? 1 : 0, 2e-12);
    }
  const auto recovered = m.inverse_transform(m.sources());
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(recovered(i, 0), d.double_at("x", i), 1e-12);
    EXPECT_NEAR(recovered(i, 1), d.double_at("y", i), 1e-12);
  }
  EXPECT_LT(m.diagnostics().double_at("whitening_error", 0), 1e-12);
  EXPECT_LT(m.diagnostics().double_at("rotation_error", 0), 1e-12);
}
TEST(ICA, FrozenQueriesRetainTrainingOriginAndRejectInvalidValues) {
  const auto d = datamunge::stats::ica_benchmark(300);
  const ICA  m(d, kMix, {3});
  const auto q   = d.take_rows({7, 2, 19});
  const auto all = m.transform(q);
  for (std::size_t i = 0; i < q.nrows(); ++i) {
    const auto one = m.transform(q.take_rows({i}));
    for (std::size_t c = 0; c < 3; ++c)
      EXPECT_DOUBLE_EQ(one(0, c), all(i, c));
  }
  auto missing = q.mutate("x1", std::vector<std::optional<double>>{1, std::nullopt, 3});
  EXPECT_THROW((void)m.transform(missing), std::invalid_argument);
  auto infinity =
      q.mutate("x1", std::vector<double>{1, std::numeric_limits<double>::infinity(), 3});
  EXPECT_THROW((void)m.transform(infinity), std::invalid_argument);
  auto text = q;
  text      = text.select({"x2", "x3"});
  text.add_column("x1", std::vector<std::string>{"a", "b", "c"});
  EXPECT_THROW((void)m.transform(text), std::invalid_argument);
  EXPECT_EQ(m.transform(q.head(0)).rows(), 0);
  EXPECT_THROW(
      (void)m.inverse_transform(datamunge::linalg::DenseMatrix<double>(2, 2)),
      std::invalid_argument);
}
TEST(ICA, EarlyExhaustionStillHasOrthonormalDirectionsAndValidPseudoInverse) {
  const auto d = datamunge::stats::ica_benchmark(400);
  const ICA  m(d, kMix, {3, "logcosh", 1, 1e-14, 8});
  EXPECT_FALSE(m.converged());
  EXPECT_EQ(m.iterations(), (std::vector<std::size_t>{1, 1, 1}));
  EXPECT_EQ(m.convergence_history().nrows(), 3);
  EXPECT_LT(m.diagnostics().double_at("rotation_error", 0), 1e-12);
  EXPECT_LT(m.diagnostics().double_at("reconstruction_rmse", 0), 1e-11);
}
TEST(ICA, RankThresholdIsRelativeAndTruncationIsAnOrthogonalProjector) {
  const auto d = datamunge::stats::ica_benchmark(200, 9, "uniform");
  for (double scale : {1e-10, 1., 1e10}) {
    DataFrame x;
    x.add_column("x", d.numeric_column("x1").values());
    x.add_column("y", d.numeric_column("x2").values());
    for (const auto& name : std::vector<std::string>{"x", "y"}) {
      auto v = x.numeric_column(name).values();
      for (auto& z : v)
        z *= scale;
      x = x.mutate(name, v);
    }
    const ICA m(x, {"x", "y"}, {2});
    EXPECT_LT(m.diagnostics().double_at("source_covariance_error", 0), 1e-11);
    const auto b = m.unmixing_matrix() * m.mixing_matrix();
    EXPECT_NEAR(b(0, 0), 1, 1e-12);
    EXPECT_NEAR(b(0, 1), 0, 1e-12);
  }
  const ICA  reduced(d, {"x1", "x2"}, {1});
  const auto projector = reduced.mixing_matrix() * reduced.unmixing_matrix();
  const auto squared   = projector * projector;
  for (std::size_t a = 0; a < 2; ++a)
    for (std::size_t b = 0; b < 2; ++b) {
      EXPECT_NEAR(projector(a, b), projector(b, a), 1e-12);
      EXPECT_NEAR(squared(a, b), projector(a, b), 1e-12);
    }
  EXPECT_GT(reduced.diagnostics().double_at("reconstruction_rmse", 0), .01);
  EXPECT_LT(reduced.diagnostics().double_at("retained_variance_fraction", 0), 1);
}
TEST(ICA, SourceMatchingUsesDistinctComponentsAndRestoresSignAndScale) {
  const datamunge::linalg::DenseMatrix<double> s(4, 2, {-1, -1, -1, 1, 1, -1, 1, 1});
  datamunge::linalg::DenseMatrix<double>       r(4, 2);
  for (std::size_t i = 0; i < 4; ++i) {
    r(i, 0) = -3 * s(i, 1) + 7;
    r(i, 1) = 2 * s(i, 0) - 5;
  }
  const auto table = datamunge::stats::ica_source_matches(s, r);
  EXPECT_EQ(table.double_at("component", 0), 2);
  EXPECT_EQ(table.double_at("component", 1), 1);
  EXPECT_EQ(table.double_at("correlation", 1), -1);
  for (std::size_t c = 0; c < 2; ++c)
    for (std::size_t i = 0; i < 4; ++i) {
      const auto j = std::size_t(table.double_at("component", c)) - 1;
      EXPECT_NEAR(
          table.double_at("offset", c) + table.double_at("multiplier", c) * r(i, j),
          s(i, c),
          1e-14);
    }
  const datamunge::linalg::DenseMatrix<double> duplicate(4, 2, {-1, -1, -1, -1, 1, 1, 1, 1});
  const auto matched = datamunge::stats::ica_source_matches(s, duplicate);
  EXPECT_NE(matched.double_at("component", 0), matched.double_at("component", 1));
  EXPECT_NEAR(
      matched.double_at("absolute_correlation", 0) + matched.double_at("absolute_correlation", 1),
      1,
      1e-14);
}
TEST(ICA, IndependentBenchmarkRecoversDistinctSources) {
  const auto                             d = datamunge::stats::ica_benchmark(3000, 4);
  const ICA                              m(d, kMix, {3, "logcosh", 500, 1e-8, 42});
  datamunge::linalg::DenseMatrix<double> s(d.nrows(), 3);
  for (std::size_t i = 0; i < d.nrows(); ++i)
    for (std::size_t j = 0; j < 3; ++j)
      s(i, j) = d.double_at("s" + std::to_string(j + 1), i);
  const auto matches = datamunge::stats::ica_source_matches(s, m.sources());
  EXPECT_TRUE(m.converged());
  for (std::size_t j = 0; j < 3; ++j)
    EXPECT_GT(matches.double_at("absolute_correlation", j), .98);
  EXPECT_EQ(datamunge::stats::ica_benchmark(20, 7, "uniform").ncols(), 5);
}
TEST(ICA, InvalidControlsRanksAndFixedPointInputsFailExplicitly) {
  const auto d = datamunge::stats::ica_benchmark(100);
  ICAOptions o;
  o.max_iter = 0;
  EXPECT_THROW(ICA(d, kMix, o), std::invalid_argument);
  for (double tol : {0., -1., 1., std::numeric_limits<double>::quiet_NaN()}) {
    o     = {};
    o.tol = tol;
    EXPECT_THROW(ICA(d, kMix, o), std::invalid_argument);
  }
  EXPECT_THROW(ICA(d, {"x1", "x1"}, {2}), std::invalid_argument);
  DataFrame constant;
  constant.add_column("x", std::vector<double>{1, 1, 1});
  EXPECT_THROW(ICA(constant, {"x"}, {1}), std::invalid_argument);
  const datamunge::linalg::DenseMatrix<double> z(4, 2, {-1, -1, -1, 1, 1, -1, 1, 1});
  EXPECT_THROW((void)datamunge::stats::ica_fixed_point_step(z, {1, 1}), std::invalid_argument);
  EXPECT_THROW(
      (void)datamunge::stats::ica_fixed_point_step(z, {1, 0}, "bogus"), std::invalid_argument);
  const datamunge::linalg::DenseMatrix<double> zero(4, 1, 0.);
  EXPECT_THROW((void)datamunge::stats::ica_fixed_point_step(zero, {1}, "cube"), std::runtime_error);
  EXPECT_THROW((void)datamunge::stats::ica_source_matches(z, zero), std::invalid_argument);
  EXPECT_THROW((void)datamunge::stats::ica_benchmark(1), std::invalid_argument);
}
