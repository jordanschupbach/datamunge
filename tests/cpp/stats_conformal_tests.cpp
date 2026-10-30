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

TEST(Conformal, QuantileAndCoverageLawByHand) {
  using datamunge::stats::conformal_coverage;
  using datamunge::stats::conformal_quantile;
  EXPECT_EQ(conformal_quantile({5, 1, 4, 2, 3, 9, 7, 8, 6}, 0.2), 8.0);  // k = ceil(10 * 0.8) = 8
  EXPECT_TRUE(std::isinf(conformal_quantile({1, 2, 3}, 0.1)));       // k = 4 > 3
  const auto c = conformal_coverage(19, 0.1);
  EXPECT_EQ(c.rank, 18u);
  EXPECT_DOUBLE_EQ(c.mean, 0.9);
  EXPECT_NEAR(c.sd, std::sqrt(18.0 * 2.0 / (400.0 * 21.0)), 1e-15);
  const auto d = conformal_coverage(20, 0.1);
  EXPECT_EQ(d.rank, 19u);
  EXPECT_NEAR(d.mean, 19.0 / 21.0, 1e-15);
  EXPECT_EQ(conformal_coverage(3, 0.1).mean, 1.0);
}

TEST(Conformal, MarginalAndConditionalCoverageFollowTheExactLaw) {
  // Continuous scores: the threshold is the k-th order statistic of n uniforms, so the coverage
  // given the calibration set, F(q) = q, is Beta(k, n + 1 - k).
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const std::size_t n = 20;
  const auto law = datamunge::stats::conformal_coverage(n, 0.1);
  double sum = 0.0, sum2 = 0.0, hits = 0.0;
  const int reps = 40000;
  for (int r = 0; r < reps; ++r) {
    std::vector<double> pred(n, 0.0), truth(n);
    for (auto& t : truth) t = u(rng);
    const SplitConformalRegressor reg(pred, truth, ConformalRegressorOptions{0.1});
    sum += reg.quantile(), sum2 += reg.quantile() * reg.quantile();
    hits += u(rng) <= reg.quantile();
  }
  const double mean = sum / reps, var = sum2 / reps - mean * mean;
  EXPECT_NEAR(mean, law.mean, 4.0 * law.sd / std::sqrt(double(reps)));
  EXPECT_NEAR(std::sqrt(var), law.sd, 0.02 * law.sd);
  EXPECT_NEAR(hits / reps, law.mean, 4.0 * std::sqrt(law.mean * (1 - law.mean) / reps));
}

TEST(SplitConformalClassifier, StringLabelsAndRowsMatchTheMatrixForm) {
  std::mt19937_64 rng(3);
  std::vector<std::vector<double>> rows;
  std::vector<std::size_t> idx;
  std::vector<std::string> names;
  DenseMatrix<double> P(60, 3);
  for (std::size_t i = 0; i < 60; ++i) {
    std::vector<double> p;
    std::size_t y = 0;
    draw_example(rng, 3, p, y);
    rows.push_back(p), idx.push_back(y), names.push_back(kABC[y]);
    for (std::size_t k = 0; k < 3; ++k) P(i, k) = p[k];
  }
  for (const std::string score : {"lac", "aps"}) {
    const SplitConformalClassifier a(P, idx, kABC, {.alpha = 0.2, .score = score});
    const SplitConformalClassifier b(rows, names, kABC, {.alpha = 0.2, .score = score});
    EXPECT_EQ(a.quantile(), b.quantile());
    EXPECT_EQ(a.calibration_scores(), b.calibration_scores());
    EXPECT_EQ(a.evaluate(P, idx).set_sizes, b.evaluate(rows, names).set_sizes);
  }
  EXPECT_THROW(SplitConformalClassifier(rows, std::vector<std::string>(60, "Z"), kABC), std::invalid_argument);
}

TEST(SplitConformalClassifier, PValuesGiveTheSameSets) {
  std::mt19937_64 rng(9);
  std::vector<std::vector<double>> cal, test;
  std::vector<std::string> cal_y;
  for (int i = 0; i < 200; ++i) {
    std::vector<double> p;
    std::size_t y = 0;
    draw_example(rng, 3, p, y);
    cal.push_back(p), cal_y.push_back(kABC[y]);
    draw_example(rng, 3, p, y);
    test.push_back(p);
  }
  for (const bool conditional : {false, true}) {
    const SplitConformalClassifier cp(cal, cal_y, kABC, {.alpha = 0.1, .class_conditional = conditional});
    for (const auto& row : test) {
      const auto in = cp.membership(row);
      const auto pv = cp.p_values(row);
      for (std::size_t k = 0; k < 3; ++k) EXPECT_EQ(in[k] == 1, pv[k] > 0.1) << conditional;
    }
  }
}

TEST(SplitConformalClassifier, ClassConditionalCoversEveryClass) {
  // Class C is hard: its probabilities are uninformative, so a marginal threshold tuned on the
  // easy classes under-covers it. Mondrian calibration restores coverage within each class.
  std::mt19937_64 rng(17);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto draw = [&](std::vector<double>& p, std::size_t& y) {
    const double r = u(rng);
    y = r < 0.475 ? 0 : (r < 0.95 ? 1 : 2);
    if (y == 2) p = {0.4, 0.3, 0.3};
    else {
      const double c = 0.6 + 0.4 * u(rng);
      p = {y == 0 ? c : (1 - c) * 0.8, y == 1 ? c : (1 - c) * 0.8, (1 - c) * 0.2};
      const double s = p[0] + p[1] + p[2];
      for (auto& v : p) v /= s;
    }
  };
  std::vector<std::vector<double>> cal, test;
  std::vector<std::string> cy, ty;
  for (int i = 0; i < 3000; ++i) {
    std::vector<double> p;
    std::size_t y = 0;
    draw(p, y), cal.push_back(p), cy.push_back(kABC[y]);
    draw(p, y), test.push_back(p), ty.push_back(kABC[y]);
  }
  const SplitConformalClassifier marginal(cal, cy, kABC, {.alpha = 0.1});
  const SplitConformalClassifier mondrian(cal, cy, kABC, {.alpha = 0.1, .class_conditional = true});
  const auto em = marginal.evaluate(test, ty), ec = mondrian.evaluate(test, ty);
  EXPECT_GE(em.coverage, 0.88);
  EXPECT_LT(em.class_coverage[2], 0.5);
  for (const double c : ec.class_coverage) EXPECT_GE(c, 0.87);
  std::size_t total = 0;
  for (const auto c : ec.size_counts) total += c;
  EXPECT_EQ(total, test.size());
}

TEST(SplitConformalRegressor, NormalisedAndQuantileBandScores) {
  // Normalised: residuals 1, 2, 3, 4 with scales 1, 2, 3, 4 all score 1.
  const auto norm = SplitConformalRegressor::normalized({0, 0, 0, 0}, {1, 2, 3, 4}, {1, -2, 3, -4}, {.alpha = 0.2});
  EXPECT_DOUBLE_EQ(norm.quantile(), 1.0);
  const auto [a, b] = norm.predict_interval(10.0, 10.0, 2.5);
  EXPECT_DOUBLE_EQ(a, 7.5);
  EXPECT_DOUBLE_EQ(b, 12.5);
  // CQR: all responses well inside [-1, 1] give negative scores, which shrink the band.
  const auto cqr = SplitConformalRegressor::quantile_band({-1, -1, -1, -1}, {1, 1, 1, 1}, {0.0, 0.5, -0.2, 0.1}, {.alpha = 0.2});
  EXPECT_DOUBLE_EQ(cqr.quantile(), -0.5);  // largest of max(-1 - y, y - 1) = -0.5 at y = 0.5
  const auto [c, d] = cqr.predict_interval(-2.0, 2.0, 1.0);
  EXPECT_DOUBLE_EQ(c, -1.5);
  EXPECT_DOUBLE_EQ(d, 1.5);
  // The general constructor with a unit scale is the absolute-residual one.
  const SplitConformalRegressor plain({1, 2, 3}, {1.5, 1.0, 3.2}, {.alpha = 0.5});
  const SplitConformalRegressor general({1, 2, 3}, {1, 2, 3}, {1, 1, 1}, {1.5, 1.0, 3.2}, {.alpha = 0.5});
  EXPECT_EQ(plain.quantile(), general.quantile());
  const auto ev = cqr.evaluate({-1, -1}, {1, 1}, {1, 1}, {0.4, 0.9});
  EXPECT_EQ(ev.covered, (std::vector<char>{1, 0}));
  EXPECT_DOUBLE_EQ(ev.average_width, 1.0);
  EXPECT_THROW(SplitConformalRegressor::normalized({0}, {0}, {1}), std::invalid_argument);
}

TEST(SplitConformalClassifier, NonemptyAddsTheTopClassOnlyToEmptySets) {
  const std::vector<std::vector<double>> cal{{0.7, 0.2, 0.1}, {0.5, 0.4, 0.1}, {0.2, 0.6, 0.2}, {0.1, 0.8, 0.1}, {0.3, 0.3, 0.4},
                                             {0.6, 0.3, 0.1}, {0.2, 0.2, 0.6}, {0.4, 0.5, 0.1}, {0.1, 0.1, 0.8}};
  const std::vector<std::string> y{"A", "B", "B", "B", "A", "A", "C", "A", "C"};
  const SplitConformalClassifier plain(cal, y, kABC, {.alpha = 0.2});
  const SplitConformalClassifier fallback(cal, y, kABC, {.alpha = 0.2, .nonempty = true});
  EXPECT_TRUE(plain.predict_set({0.35, 0.33, 0.32}).empty());  // threshold 0.6: every score is above it
  EXPECT_EQ(fallback.predict_set({0.35, 0.33, 0.32}), (std::vector<std::string>{"A"}));
  EXPECT_EQ(fallback.predict_set({0.45, 0.45, 0.10}), plain.predict_set({0.45, 0.45, 0.10}));
}
