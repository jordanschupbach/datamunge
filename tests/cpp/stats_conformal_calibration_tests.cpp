#include <datamunge/stats/conformal.hpp>
#include <datamunge/stats/conformal_calibration.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace datamunge::stats;

TEST(CrossConformal, CvPlusPairsResidualsWithTheirFoldAndBatchesTargets) {
  const std::vector<double> truth{2, 6, 4, 10, 8, 12};
  const std::vector<std::size_t> labels{12, 7, 12, 7, 12, 7};
  const std::vector<std::size_t> even{0, 2, 4}, odd{1, 3, 5};
  std::size_t refits = 0;
  const auto intervals = cv_plus_intervals(truth, labels, [&](const auto& train, const auto& held) {
    ++refits;
    if (held == even) {
      EXPECT_EQ(train, odd);
      EXPECT_EQ(refits, 1u);
      return ConformalFoldPredictions{{1, 6, 5}, {10, -10}};
    }
    EXPECT_EQ(train, even);
    EXPECT_EQ(held, odd);
    return ConformalFoldPredictions{{2, 5, 9}, {20, -20}};
  }, .3);
  EXPECT_EQ(refits, 2u);
  ASSERT_EQ(intervals.size(), 2u);
  EXPECT_EQ(intervals[0], (std::pair{8.0, 24.0}));
  EXPECT_EQ(intervals[1], (std::pair{-24.0, -8.0}));
}

TEST(CrossConformal, JackknifePlusUsesLeaveOneOutFitsAndFiniteSampleRanks) {
  const std::vector<double> truth{1, 3, 5, 7};
  std::size_t refits = 0;
  const auto refit = [&](const auto& train, const auto& held) {
    EXPECT_EQ(train.size(), 3u);
    EXPECT_EQ(held.size(), 1u);
    double mean = 0;
    for (const auto row : train) {
      EXPECT_NE(row, held.front());
      mean += truth[row] / 3;
    }
    ++refits;
    return ConformalFoldPredictions{{mean}, {mean}};
  };
  const auto bands = jackknife_plus_intervals(truth, refit, .25);
  ASSERT_EQ(bands.size(), 1u);
  EXPECT_NEAR(bands.front().first, -1, 1e-15);
  EXPECT_NEAR(bands.front().second, 9, 1e-15);
  EXPECT_EQ(refits, truth.size());

  const auto unrestricted = jackknife_plus_intervals(truth, refit, .1).front();
  const auto inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(unrestricted, (std::pair{-inf, inf}));
  EXPECT_THROW(cv_plus_intervals(truth, {0, 0, 0, 0}, refit), std::invalid_argument);
  EXPECT_THROW(cv_plus_intervals(truth, {0, 0, 1, 1}, [](const auto&, const auto&) {
    return ConformalFoldPredictions{{1}, {1}};
  }), std::invalid_argument);
}

TEST(WeightedConformal, ThresholdIncludesTestMassAndIgnoresZeroWeightScores) {
  const std::vector<double> scores{3, 1, 2, 3, 50}, weights{1, 2, 1, 2, 0};
  const WeightedConformalCalibration fit(scores, weights);
  EXPECT_EQ(fit.threshold(0, .5), 2);
  EXPECT_EQ(fit.threshold(2, .5), 3);
  EXPECT_EQ(fit.threshold(2, .25), 3); // Equality at the exact threshold stays finite.
  EXPECT_TRUE(std::isinf(fit.threshold(3, .25)));

  const auto unit = std::numeric_limits<double>::max() / 2;
  const WeightedConformalCalibration huge(scores, {unit, 2 * unit, unit, 2 * unit, 0});
  EXPECT_EQ(huge.threshold(2 * unit, .25), fit.threshold(2, .25));
  EXPECT_EQ(huge.threshold(0, .5), fit.threshold(0, .5));
  const WeightedConformalCalibration zero({-1}, {0});
  EXPECT_TRUE(std::isinf(zero.threshold(1)));
  EXPECT_THROW(zero.threshold(0), std::invalid_argument);
  EXPECT_THROW(fit.threshold(-1, .25), std::invalid_argument);
  EXPECT_THROW((WeightedConformalCalibration{{1, 2}, {1}}), std::invalid_argument);
}

TEST(WeightedConformal, EqualWeightsMatchSplitConformalRanks) {
  const std::vector<double> scores{2, -4, 0, 1, 9, 12, 14, 19, 30};
  const WeightedConformalCalibration fit(scores, std::vector<double>(scores.size(), 7));
  for (const double alpha : {.05, .1, .2, .25, .5, .75})
    EXPECT_EQ(fit.threshold(7, alpha), conformal_quantile(scores, alpha));
}

} // namespace
