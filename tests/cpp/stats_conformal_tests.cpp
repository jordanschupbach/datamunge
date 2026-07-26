#include <gtest/gtest.h>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/conformal.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using datamunge::linalg::DenseMatrix;
using datamunge::stats::ConformalClassifierOptions;
using datamunge::stats::ConformalRegressorOptions;
using datamunge::stats::SplitConformalClassifier;
using datamunge::stats::SplitConformalRegressor;

namespace {

const std::vector<std::string> kABC = {"A", "B", "C"};

// A random probability vector over K classes, and a label drawn from it (so calibration and test
// examples are i.i.d., hence exchangeable -- the only assumption conformal prediction needs).
void draw_example(std::mt19937_64& rng, std::size_t K, std::vector<double>& probs, std::size_t& label) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  probs.assign(K, 0.0);
  double sum = 0.0;
  for (std::size_t k = 0; k < K; ++k) { probs[k] = -std::log(u(rng) + 1e-12); sum += probs[k]; }
  for (double& p : probs) p /= sum;
  double r = u(rng), c = 0.0;
  label = K - 1;
  for (std::size_t k = 0; k < K; ++k) { c += probs[k]; if (r <= c) { label = k; break; } }
}

} // namespace

TEST(SplitConformalClassifier, CoverageGuaranteeHolds) {
  // Over many trials with fresh exchangeable calibration+test data, the average coverage of the
  // conformal set must meet the 1 - alpha target (the marginal guarantee).
  std::mt19937_64 rng(12345);
  const std::size_t K = 3, n_cal = 200;
  for (const std::string& sc : {"lac", "aps"}) {
    for (double alpha : {0.2, 0.1, 0.05}) {
      const int T = 4000;
      std::size_t covered = 0;
      for (int t = 0; t < T; ++t) {
        DenseMatrix<double> cal(n_cal, K, 0.0);
        std::vector<std::size_t> lab(n_cal);
        std::vector<double> p;
        std::size_t y;
        for (std::size_t i = 0; i < n_cal; ++i) {
          draw_example(rng, K, p, y);
          for (std::size_t k = 0; k < K; ++k) cal(i, k) = p[k];
          lab[i] = y;
        }
        SplitConformalClassifier conf(cal, lab, kABC, ConformalClassifierOptions{alpha, sc});
        draw_example(rng, K, p, y);  // one fresh test point
        if (conf.membership(p)[y]) ++covered;
      }
      const double coverage = static_cast<double>(covered) / T;
      EXPECT_GE(coverage, 1.0 - alpha - 0.02) << "score=" << sc << " alpha=" << alpha << " cov=" << coverage;
    }
  }
}

TEST(SplitConformalClassifier, QuantileMatchesFiniteSampleRank) {
  // Calibration LAC scores 0.1, 0.2, ..., 0.9 (true-class probs 0.9 down to 0.1).
  const std::size_t n = 9, K = 3;
  DenseMatrix<double> cal(n, K, 0.0);
  std::vector<std::size_t> lab(n, 0);  // true class = A for all
  for (std::size_t i = 0; i < n; ++i) {
    const double pa = 0.9 - 0.1 * static_cast<double>(i);  // 0.9 .. 0.1
    cal(i, 0) = pa;
    cal(i, 1) = (1.0 - pa) / 2.0;
    cal(i, 2) = (1.0 - pa) / 2.0;
  }
  // alpha=0.1: rank = ceil(10 * 0.9) = 9 -> 9th smallest score = 0.9.
  SplitConformalClassifier a(cal, lab, kABC, ConformalClassifierOptions{0.1, "lac"});
  EXPECT_NEAR(a.quantile(), 0.9, 1e-9);
  // alpha=0.2: rank = ceil(10 * 0.8) = 8 -> 8th smallest = 0.8.
  SplitConformalClassifier b(cal, lab, kABC, ConformalClassifierOptions{0.2, "lac"});
  EXPECT_NEAR(b.quantile(), 0.8, 1e-9);

  // A test point with p = (0.5, 0.3, 0.2): LAC set at q=0.8 is {k : p_k >= 1 - 0.8 = 0.2} = {A,B,C}.
  const auto set = b.predict_set({0.5, 0.3, 0.2});
  EXPECT_EQ(set.size(), 3u);
  // At a stricter q=0.9 (alpha=0.1) it is {k : p_k >= 0.1} = still all three here.
  EXPECT_EQ(a.predict_set({0.5, 0.3, 0.2}).size(), 3u);
  // Larger sets for smaller alpha (monotonicity of the threshold).
  EXPECT_GE(a.quantile(), b.quantile());
}

TEST(SplitConformalClassifier, TinyCalibrationForcesFullSets) {
  // With n=4 and alpha=0.05, rank = ceil(5 * 0.95) = 5 > 4, so the guarantee forces all-inclusive sets.
  DenseMatrix<double> cal(4, 3, 0.0);
  std::vector<std::size_t> lab = {0, 1, 2, 0};
  for (std::size_t i = 0; i < 4; ++i) { cal(i, lab[i]) = 0.8; for (std::size_t k = 0; k < 3; ++k) if (k != lab[i]) cal(i, k) = 0.1; }
  SplitConformalClassifier conf(cal, lab, kABC, ConformalClassifierOptions{0.05, "lac"});
  EXPECT_EQ(conf.predict_set({0.98, 0.01, 0.01}).size(), 3u);  // even a confident point gets the full set
}

TEST(SplitConformalRegressor, CoverageGuaranteeHolds) {
  std::mt19937_64 rng(999);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.0, 10.0);
  const std::size_t n_cal = 200;
  for (double alpha : {0.2, 0.1, 0.05}) {
    const int T = 4000;
    std::size_t covered = 0;
    for (int t = 0; t < T; ++t) {
      std::vector<double> pred(n_cal), truth(n_cal);
      for (std::size_t i = 0; i < n_cal; ++i) { pred[i] = u(rng); truth[i] = pred[i] + noise(rng); }
      SplitConformalRegressor reg(pred, truth, ConformalRegressorOptions{alpha});
      const double tp = u(rng), tt = tp + noise(rng);
      const auto [lo, hi] = reg.predict_interval(tp);
      if (tt >= lo && tt <= hi) ++covered;
    }
    const double coverage = static_cast<double>(covered) / T;
    EXPECT_GE(coverage, 1.0 - alpha - 0.02) << "alpha=" << alpha << " cov=" << coverage;
  }
}

TEST(SplitConformalRegressor, IntervalIsResidualQuantile) {
  // Residuals 0,1,2,...,9. alpha=0.1: rank = ceil(11*0.9) = 10 -> 10th smallest = 9.
  std::vector<double> pred(10, 0.0), truth(10);
  for (std::size_t i = 0; i < 10; ++i) truth[i] = static_cast<double>(i);  // residual = i
  SplitConformalRegressor reg(pred, truth, ConformalRegressorOptions{0.1});
  EXPECT_NEAR(reg.quantile(), 9.0, 1e-9);
  const auto [lo, hi] = reg.predict_interval(5.0);
  EXPECT_NEAR(lo, -4.0, 1e-9);
  EXPECT_NEAR(hi, 14.0, 1e-9);
}

TEST(Conformal, RejectsInvalidOptions) {
  DenseMatrix<double> cal(3, 3, 0.3);
  std::vector<std::size_t> lab = {0, 1, 2};
  EXPECT_THROW(SplitConformalClassifier(cal, lab, kABC, ConformalClassifierOptions{0.0, "lac"}), std::invalid_argument);
  EXPECT_THROW(SplitConformalClassifier(cal, lab, kABC, ConformalClassifierOptions{1.0, "lac"}), std::invalid_argument);
  EXPECT_THROW(SplitConformalClassifier(cal, lab, kABC, ConformalClassifierOptions{0.1, "bad"}), std::invalid_argument);
  std::vector<std::size_t> short_lab = {0, 1};
  EXPECT_THROW(SplitConformalClassifier(cal, short_lab, kABC, ConformalClassifierOptions{0.1, "lac"}),
               std::invalid_argument);
  EXPECT_THROW(SplitConformalRegressor({1.0, 2.0}, {1.0}, ConformalRegressorOptions{0.1}), std::invalid_argument);
}
