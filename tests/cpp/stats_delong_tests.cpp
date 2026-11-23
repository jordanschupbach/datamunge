#include <datamunge/nn/metrics.hpp>
#include <datamunge/stats/delong.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using namespace datamunge::stats;

TEST(DeLongAuc, MatchesDirectPairPlacementsWithTiesAndUnequalClassSizes) {
  const std::vector<int> truth{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  const std::vector<double> first{2, 7, 3, 1, 5, 6, 4, 3, 6, 5, 4};
  const std::vector<double> second{3, 7, 2, 4, 5, 6, 1, 2, 6, 5, 1};
  const std::vector<double> reversed{-2, -7, -3, -1, -5, -6, -4, -3, -6, -5, -4};
  const auto fit = delong_auc_covariance(truth, {first, second, reversed});
  ASSERT_EQ(fit.areas.size(), 3u);
  EXPECT_DOUBLE_EQ(fit.areas[0], datamunge::nn::roc_auc(truth, first));
  EXPECT_DOUBLE_EQ(fit.areas[1], datamunge::nn::roc_auc(truth, second));
  EXPECT_DOUBLE_EQ(fit.areas[2], datamunge::nn::roc_auc(truth, reversed));
  EXPECT_NEAR(fit.covariance(0, 0), .025166666666666664, 1e-15);
  EXPECT_NEAR(fit.covariance(0, 1), .023111111111111107, 1e-15);
  EXPECT_NEAR(fit.covariance(1, 1), .040277777777777773, 1e-15);
  EXPECT_DOUBLE_EQ(fit.covariance(1, 0), fit.covariance(0, 1));
  EXPECT_DOUBLE_EQ(fit.covariance(0, 2), -fit.covariance(0, 0));
  EXPECT_DOUBLE_EQ(fit.covariance(1, 2), -fit.covariance(1, 0));
  EXPECT_DOUBLE_EQ(fit.covariance(2, 2), fit.covariance(0, 0));

  const auto test = delong_auc_test(truth, first, second);
  EXPECT_DOUBLE_EQ(test.estimate1, fit.areas[0]);
  EXPECT_DOUBLE_EQ(test.estimate2, fit.areas[1]);
  EXPECT_NEAR(test.statistic, .96169407366868953, 1e-14);
  EXPECT_NEAR(test.p_value, .33620330016389643, 1e-14);
  ASSERT_TRUE(test.has_conf_int);
  EXPECT_NEAR(test.conf_int_lower, -.13840436208064161, 1e-8);
  EXPECT_NEAR(test.conf_int_upper, .40507102874730821, 1e-8);
  const auto swapped = delong_auc_test(truth, second, first);
  EXPECT_DOUBLE_EQ(swapped.statistic, -test.statistic);
  EXPECT_DOUBLE_EQ(swapped.p_value, test.p_value);
  EXPECT_DOUBLE_EQ(swapped.conf_int_lower, -test.conf_int_upper);
}

TEST(DeLongAuc, HandlesTiedAndDeterministicContrastsAndValidatesInputs) {
  const std::vector<int> truth{0, 0, 1, 1};
  const std::vector<double> tied{3, 3, 3, 3};
  const std::vector<double> first{0, 0, 1, 1};
  const std::vector<double> second{1, 1, 0, 0};
  const auto auc = delong_auc_covariance(truth, {tied});
  EXPECT_DOUBLE_EQ(auc.areas[0], .5);
  EXPECT_DOUBLE_EQ(auc.covariance(0, 0), 0);

  const auto identical = delong_auc_test(truth, {0, 1, 1, 2}, {3, 5, 5, 9});
  EXPECT_DOUBLE_EQ(identical.statistic, 0);
  EXPECT_DOUBLE_EQ(identical.p_value, 1);
  EXPECT_DOUBLE_EQ(identical.conf_int_lower, 0);
  EXPECT_DOUBLE_EQ(identical.conf_int_upper, 0);
  const auto extreme = delong_auc_test(truth, second, first);
  EXPECT_EQ(extreme.statistic, -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(extreme.p_value, 0);
  EXPECT_DOUBLE_EQ(extreme.conf_int_lower, -1);
  EXPECT_DOUBLE_EQ(extreme.conf_int_upper, -1);
  EXPECT_THROW(delong_auc_covariance(truth, {}), std::invalid_argument);
  EXPECT_THROW(delong_auc_covariance({1, 0, 0, 0}, {tied}), std::invalid_argument);
  EXPECT_THROW(delong_auc_covariance({2, 0, 1, 1}, {tied}), std::invalid_argument);
  EXPECT_THROW(delong_auc_test(truth, {0, 1, 2}, tied), std::invalid_argument);
  EXPECT_THROW(delong_auc_test(truth, {0, 1, 2, std::nan("")}, tied), std::invalid_argument);
  EXPECT_THROW(delong_auc_test(truth, first, second, 1), std::invalid_argument);
}

} // namespace
