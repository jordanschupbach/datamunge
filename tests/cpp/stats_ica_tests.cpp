#include <gtest/gtest.h>

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/ica.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::ICA;
using datamunge::stats::ICAOptions;

namespace {

const std::vector<std::string> kMix = {"x1", "x2", "x3"};

double abscorr(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = a.size();
  double ma = 0.0, mb = 0.0;
  for (std::size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  double num = 0.0, da = 0.0, db = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xa = a[i] - ma, xb = b[i] - mb;
    num += xa * xb; da += xa * xa; db += xb * xb;
  }
  return std::abs(num / std::sqrt(da * db));
}

// Three deterministic non-Gaussian sources (sine, square, sawtooth) linearly mixed into three
// observed signals; the true sources are returned alongside for recovery scoring.
DataFrame make_mixtures(std::vector<std::vector<double>>& sources_true, std::size_t n = 1000) {
  std::vector<double> s1(n), s2(n), s3(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i);
    s1[i] = std::sin(0.060 * t);
    s2[i] = (std::sin(0.093 * t) >= 0.0) ? 1.0 : -1.0;
    s3[i] = std::fmod(0.050 * t, 2.0) - 1.0;
  }
  sources_true = {s1, s2, s3};
  const double A[3][3] = {{0.80, 0.30, 0.20}, {0.40, 0.90, 0.50}, {0.60, 0.20, 0.70}};
  std::vector<double> x1(n), x2(n), x3(n);
  for (std::size_t i = 0; i < n; ++i) {
    x1[i] = A[0][0] * s1[i] + A[0][1] * s2[i] + A[0][2] * s3[i];
    x2[i] = A[1][0] * s1[i] + A[1][1] * s2[i] + A[1][2] * s3[i];
    x3[i] = A[2][0] * s1[i] + A[2][1] * s2[i] + A[2][2] * s3[i];
  }
  return DataFrame{{"x1", DataFrame::numeric_column_type(x1)},
                   {"x2", DataFrame::numeric_column_type(x2)},
                   {"x3", DataFrame::numeric_column_type(x3)}};
}

double mean_best_recovery(const std::vector<std::vector<double>>& truth, const ICA& ica) {
  double acc = 0.0;
  for (const auto& s : truth) {
    double best = 0.0;
    for (std::size_t c = 0; c < ica.n_components(); ++c) best = std::max(best, abscorr(s, ica.component(c)));
    acc += best;
  }
  return acc / static_cast<double>(truth.size());
}

} // namespace

TEST(ICA, RecoversMixedSources) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  ICA ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  EXPECT_EQ(ica.observations(), 1000u);
  EXPECT_EQ(ica.n_components(), 3u);
  EXPECT_TRUE(ica.converged());
  EXPECT_GT(mean_best_recovery(truth, ica), 0.95);
}

TEST(ICA, AllContrastsRecover) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  for (const std::string& g : {"logcosh", "exp", "cube"}) {
    ICA ica(mixed, kMix, ICAOptions{3, g, 300, 1e-6, 42});
    EXPECT_GT(mean_best_recovery(truth, ica), 0.90) << "contrast " << g;
  }
}

TEST(ICA, SourcesAreStandardizedAndDecorrelated) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  ICA ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  const std::size_t n = ica.observations();
  for (std::size_t c = 0; c < 3; ++c) {
    double m = 0.0, ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) m += ica.sources()(i, c);
    m /= n;
    for (std::size_t i = 0; i < n; ++i) ss += (ica.sources()(i, c) - m) * (ica.sources()(i, c) - m);
    EXPECT_NEAR(m, 0.0, 1e-6);
    EXPECT_NEAR(ss / (n - 1), 1.0, 1e-2);  // unit variance by whitening
  }
  // Distinct recovered components are (at least) decorrelated.
  EXPECT_LT(abscorr(ica.component(0), ica.component(1)), 1e-3);
  EXPECT_LT(abscorr(ica.component(0), ica.component(2)), 1e-3);
  EXPECT_LT(abscorr(ica.component(1), ica.component(2)), 1e-3);
}

TEST(ICA, TransformReproducesSourcesOnTrainingData) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  ICA ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  const auto t = ica.transform(mixed);
  ASSERT_EQ(t.rows(), ica.observations());
  ASSERT_EQ(t.cols(), 3u);
  for (std::size_t i = 0; i < ica.observations(); ++i)
    for (std::size_t c = 0; c < 3; ++c) EXPECT_NEAR(t(i, c), ica.sources()(i, c), 1e-9);
}

TEST(ICA, MixingMatrixReconstructsTheMixtures) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  ICA ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  // Centered mixtures should equal sources * mixing^T (full-rank, 3 components from 3 features).
  double ss = 0.0;
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < ica.observations(); ++i)
    for (std::size_t l = 0; l < 3; ++l) {
      double recon = 0.0;
      for (std::size_t c = 0; c < 3; ++c) recon += ica.sources()(i, c) * ica.mixing_matrix()(l, c);
      const double centered = mixed.double_at(kMix[l], i) - ica.mean()[l];
      ss += (centered - recon) * (centered - recon);
      ++cnt;
    }
  EXPECT_LT(std::sqrt(ss / cnt), 1e-6);
}

TEST(ICA, DeterministicGivenSeed) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  ICA a(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  ICA b(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  for (std::size_t i = 0; i < a.observations(); ++i)
    for (std::size_t c = 0; c < 3; ++c) EXPECT_NEAR(a.sources()(i, c), b.sources()(i, c), 1e-12);
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
  const auto mixed = make_mixtures(truth, 200);
  EXPECT_THROW(ICA(mixed, {}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, kMix, ICAOptions{0, "logcosh", 200, 1e-6, 42}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, kMix, ICAOptions{4, "logcosh", 200, 1e-6, 42}), std::invalid_argument);
  EXPECT_THROW(ICA(mixed, kMix, ICAOptions{3, "bogus", 200, 1e-6, 42}), std::invalid_argument);
}

TEST(ICA, PlotAndAccessorsGuardRanges) {
  std::vector<std::vector<double>> truth;
  const auto mixed = make_mixtures(truth);
  ICA ica(mixed, kMix, ICAOptions{3, "logcosh", 200, 1e-6, 42});
  std::vector<std::string> labels(ica.observations(), "grp");
  EXPECT_EQ(ica.plot_sources(labels).series().size(), 1u);
  EXPECT_THROW(ica.plot_sources(std::vector<std::string>{"too-few"}), std::invalid_argument);
  EXPECT_THROW(ica.plot_sources(labels, 0, 9), std::out_of_range);
  EXPECT_THROW(ica.component(9), std::out_of_range);
}
