#include <datamunge/algorithms/adaboost.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/adaboost_classifier.hpp>
#include <datamunge/stats/train_test_split.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace stats = datamunge::stats;
namespace alg = datamunge::algorithms;

namespace {

/// The original O(n^2)-per-feature stump search, as an oracle for the sweep.
double brute_force_stump_error(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                               const std::vector<double>& w) {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < X.front().size(); ++j) {
    std::vector<double> vals;
    for (const auto& row : X) vals.push_back(row[j]);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    std::vector<double> thresholds{vals.front() - 1.0};
    for (std::size_t k = 0; k + 1 < vals.size(); ++k) thresholds.push_back(0.5 * (vals[k] + vals[k + 1]));
    thresholds.push_back(vals.back());
    for (double thr : thresholds)
      for (int pol : {1, -1}) {
        double err = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) err += ((X[i][j] <= thr) ? pol : -pol) != y[i] ? w[i] : 0.0;
        best = std::min(best, err);
      }
  }
  return best;
}

} // namespace

TEST(AdaBoostStump, SweepMatchesBruteForce) {
  datamunge::random::Pcg32 rng(4);
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<std::vector<double>> X(60, std::vector<double>(3));
    std::vector<int> y(60);
    std::vector<double> w(60);
    for (std::size_t i = 0; i < 60; ++i) {
      for (auto& v : X[i]) v = std::round(rng.uniform(0, 10));  // many ties
      y[i] = rng.bernoulli(0.4) ? 1 : -1;
      w[i] = rng.uniform(0.1, 1.0);
    }
    double err = 0.0;
    const auto stump = alg::detail::train_stump(X, y, w, err);
    EXPECT_NEAR(err, brute_force_stump_error(X, y, w), 1e-12);
    double check = 0.0;
    for (std::size_t i = 0; i < 60; ++i) check += stump.classify(X[i]) != y[i] ? w[i] : 0.0;
    EXPECT_NEAR(check, err, 1e-12);
  }
}

TEST(TrainTestSplit, StratifiedDisjointAndReproducible) {
  const auto data = datamunge::datasets::breast_cancer();
  const auto [train, test] = stats::train_test_split(data, 0.3, 1, "diagnosis");
  EXPECT_EQ(train.nrows() + test.nrows(), data.nrows());
  const auto count = [](const auto& frame, const std::string& level) {
    const auto& v = frame.template get_column<std::string>("diagnosis").values();
    return std::count(v.begin(), v.end(), level);
  };
  EXPECT_EQ(count(test, "malignant"), std::lround(0.3 * 212));
  EXPECT_EQ(count(test, "benign"), std::lround(0.3 * 357));
  const auto [train2, test2] = stats::train_test_split(data, 0.3, 1, "diagnosis");
  EXPECT_EQ(test.get_column<double>("mean_radius").values(), test2.get_column<double>("mean_radius").values());
  const auto [a, b] = stats::train_test_split(data, 0.5, 9);
  EXPECT_EQ(a.nrows() + b.nrows(), data.nrows());
  EXPECT_THROW((void)stats::train_test_split(data, 1.0), std::invalid_argument);
  EXPECT_THROW((void)stats::train_test_split(data, 0.3, 1, "mean_radius"), std::invalid_argument);
  EXPECT_THROW((void)stats::train_test_split(data, 0.3, 1, "nope"), std::invalid_argument);
}

TEST(AdaBoostClassifier, FitsBreastCancerAndHonoursTheBound) {
  const auto [train, test] = stats::train_test_split(datamunge::datasets::breast_cancer(), 0.3, 1, "diagnosis");
  const stats::AdaBoostClassifier model(train, "diagnosis ~ .", {.n_rounds = 60});
  EXPECT_EQ(model.positive_class(), "malignant");
  EXPECT_EQ(model.predictor_names().size(), 30U);
  EXPECT_EQ(model.observations(), train.nrows());

  const auto h = model.history();
  ASSERT_EQ(h.nrows(), model.n_rounds());
  for (std::size_t t = 0; t < h.nrows(); ++t) {
    EXPECT_LT(h.double_at("weak_error", t), 0.5);
    EXPECT_LE(h.double_at("train_error", t), h.double_at("bound", t) + 1e-12);
  }
  const auto staged = model.staged_error(train);
  EXPECT_NEAR(staged.back(), model.error(train), 1e-12);
  EXPECT_NEAR(staged.front(), h.double_at("train_error", 0), 1e-12);
  EXPECT_LT(model.error(test), 0.08);

  const auto margins = model.margins(test);
  const auto predicted = model.predict(test);
  const auto& truth = test.get_column<std::string>("diagnosis").values();
  for (std::size_t i = 0; i < margins.size(); ++i) {
    EXPECT_GE(margins[i], -1.0 - 1e-12);
    EXPECT_LE(margins[i], 1.0 + 1e-12);
    EXPECT_EQ(margins[i] >= 0.0, predicted[i] == truth[i]) << i;
  }
  const auto imp = model.feature_importance();
  double total = 0.0;
  for (double v : imp.get_column<double>("importance").values()) total += v;
  EXPECT_NEAR(total, 1.0, 1e-12);
  EXPECT_THROW((void)model.plot_decision_regions(test), std::logic_error);
}

TEST(AdaBoostClassifier, TwoPredictorRegionsAndErrors) {
  const auto data = datamunge::datasets::breast_cancer();
  const stats::AdaBoostClassifier model(data, "diagnosis ~ mean_radius + mean_concave_points", {.n_rounds = 10});
  const auto plot = model.plot_decision_regions(data);
  EXPECT_GT(plot.series().size(), 2U);
  EXPECT_THROW(stats::AdaBoostClassifier(data, "diagnosis ~ .", {.n_rounds = 0}), std::invalid_argument);
  EXPECT_THROW(stats::AdaBoostClassifier(datamunge::datasets::wine(), "cultivar ~ .", {}), std::invalid_argument);
}
