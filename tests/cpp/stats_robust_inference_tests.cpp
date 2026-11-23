#include <datamunge/stats/classical_tests.hpp>
#include <datamunge/stats/normality_tests.hpp>
#include <datamunge/stats/robust_tests.hpp>
#include <datamunge/stats/t_test.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace datamunge::stats;

const std::vector<double> x{1, 2, 2, 3, 3, 4, 4, 5, 8, 9, 60};
const std::vector<double> y{-100, -2, 0, 0, 1, 2, 3, 3, 4, 5, 6, 9, 15};

TEST(RobustInference, YuenMatchesSciPyAndItsZeroTrimMatchesWelch) {
  const auto fit = yuen_test(x, y);
  EXPECT_NEAR(fit.statistic, .9430725241464936, 1e-13);
  EXPECT_NEAR(fit.parameter1, 12.651755935857375, 1e-12);
  EXPECT_NEAR(fit.p_value, .3633049738330239, 1e-13);
  EXPECT_NEAR(fit.estimate1 - fit.estimate2, 1.476190476190476, 1e-14);
  ASSERT_TRUE(fit.has_conf_int);
  EXPECT_NEAR(fit.conf_int_lower, -1.9149187300985764, 1e-10);
  EXPECT_NEAR(fit.conf_int_upper, 4.86729968247953, 1e-10);

  const auto greater = yuen_test(x, y, .2, Alternative::Greater);
  EXPECT_NEAR(greater.p_value, fit.p_value / 2, 1e-15);
  EXPECT_LT(fit.conf_int_lower, greater.conf_int_lower);
  EXPECT_EQ(greater.conf_int_upper, std::numeric_limits<double>::infinity());

  const auto zero = yuen_test(x, y, 0);
  const auto welch = t_test_two_sample(x, y, false);
  EXPECT_NEAR(zero.statistic, welch.statistic, 1e-14);
  EXPECT_NEAR(zero.parameter1, welch.parameter1, 1e-13);
  EXPECT_NEAR(zero.conf_int_upper, welch.conf_int_upper, 1e-11);
  EXPECT_THROW(yuen_test({1, 2, 3}, y, .4), std::invalid_argument);
  EXPECT_THROW(yuen_test({1, 1}, {2, 2}), std::invalid_argument);
}

TEST(RobustInference, BrunnerMunzelAndCliffsDeltaHandleTiesAndDirection) {
  const auto fit = brunner_munzel_test(x, y);
  // SciPy uses the opposite sign for W, and the same direction for its p-values.
  EXPECT_NEAR(fit.statistic, 1.204458727773962, 1e-14);
  EXPECT_NEAR(fit.p_value, .2424621820021901, 1e-14);
  EXPECT_NEAR(fit.parameter1, 20.01554779477333, 1e-13);
  EXPECT_NEAR(fit.estimate1, mann_whitney_test(x, y).estimate1, 1e-15);
  EXPECT_NEAR(brunner_munzel_test(x, y, Alternative::Greater).p_value, .12123109100109505, 1e-14);
  EXPECT_NEAR(brunner_munzel_test(y, x, Alternative::Less).p_value, .12123109100109505, 1e-14);

  const auto delta = cliffs_delta(x, y);
  EXPECT_NEAR(delta.estimate, .2797202797202798, 1e-14);
  EXPECT_NEAR(delta.standard_error, .23223733057026272, 1e-14);
  ASSERT_TRUE(delta.confidence_interval);
  EXPECT_NEAR(delta.confidence_interval->first, -.20469418017896862, 1e-11);
  EXPECT_NEAR(delta.confidence_interval->second, .7641347396195282, 1e-11);
  EXPECT_NEAR(cliffs_delta(y, x).estimate, -delta.estimate, 1e-15);

  const auto separated = cliffs_delta({9, 10}, {1, 2});
  EXPECT_EQ(separated.estimate, 1);
  EXPECT_FALSE(separated.confidence_interval);
  const auto tied = cliffs_delta({1, 1, 1}, {1, 1, 1, 1, 1, 1, 1});
  EXPECT_EQ(tied.estimate, 0);
  EXPECT_EQ(tied.standard_error, 0);
  EXPECT_FALSE(tied.confidence_interval);
  EXPECT_THROW(brunner_munzel_test({1, 1, 1}, {1, 1, 1}), std::invalid_argument);
}

TEST(MomentNormality, BothOmnibusStatisticsAndComponentsMatchSciPy) {
  const std::vector<double> values{1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 8, 13, 21, 34, 55, 89, 144, 233};
  const auto dp = dagostino_pearson_test(values);
  EXPECT_NEAR(dp.statistic, 26.354452692595878, 1e-12);
  EXPECT_NEAR(dp.p_value / 1.89322970949444e-6, 1, 1e-13);
  EXPECT_NEAR(dp.skewness_contribution, 15.877939094611454, 1e-12);
  EXPECT_NEAR(dp.kurtosis_contribution, 10.476513597984425, 1e-12);
  const auto jb = jarque_bera_test(values);
  EXPECT_NEAR(jb.statistic, 38.93247389519857, 1e-12);
  EXPECT_NEAR(jb.p_value / 3.5149626149558288e-9, 1, 1e-13);
  EXPECT_NEAR(jb.skewness_contribution, 19.043230692606524, 1e-12);
  EXPECT_NEAR(jb.kurtosis_contribution, 19.88924320259208, 1e-12);

  EXPECT_EQ(dagostino_pearson_test({1, 2, 3, 4, 5, 6, 7, 8}).skewness_contribution, 0);
  EXPECT_THROW(dagostino_pearson_test({1, 2, 3, 4, 5, 6, 7}), std::invalid_argument);
  EXPECT_THROW(jarque_bera_test({1, 1, 1}), std::invalid_argument);
}

} // namespace
