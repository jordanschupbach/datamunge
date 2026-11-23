#include <datamunge/stats/standardized_effects.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using namespace datamunge::stats;

TEST(StandardizedEffects, MatchesRPointEstimatesAndConfidenceLimitsForAllStandardizers) {
  const std::vector<double> first{13, 17, 21, 19, 16, 22}, second{11, 16, 17, 15, 12};
  const auto single = one_sample_standardized_mean_difference(first, 15);
  EXPECT_EQ(single.standardization, MeanDifferenceStandardization::SampleSd);
  EXPECT_EQ(single.degrees_of_freedom, 5u);
  EXPECT_NEAR(single.standardizer, 3.346640106136302251, 1e-14);
  EXPECT_NEAR(single.cohen_d, .896421457000795230, 1e-14);
  ASSERT_TRUE(single.hedges_g);
  EXPECT_NEAR(*single.hedges_g, .753665158902013577, 1e-14);
  EXPECT_NEAR(single.confidence_interval.first, -.097763134548816272, 1e-9);
  EXPECT_NEAR(single.confidence_interval.second, 1.832950229414096910, 1e-9);

  const auto paired = paired_standardized_mean_difference({13, 17, 21, 19, 16}, second, .90);
  EXPECT_EQ(paired.standardization, MeanDifferenceStandardization::PairedDifferenceSd);
  EXPECT_EQ(paired.degrees_of_freedom, 4u);
  EXPECT_NEAR(paired.standardizer, std::sqrt(2.), 1e-14);
  EXPECT_NEAR(paired.cohen_d, 2.12132034355964239, 1e-14);
  ASSERT_TRUE(paired.hedges_g);
  EXPECT_NEAR(*paired.hedges_g, 1.69256875064326850, 1e-14);
  EXPECT_NEAR(paired.confidence_interval.first, .65999425915185361, 1e-9);
  EXPECT_NEAR(paired.confidence_interval.second, 3.45436496086534195, 1e-9);

  const auto pooled = pooled_standardized_mean_difference(first, second);
  EXPECT_EQ(pooled.standardization, MeanDifferenceStandardization::PooledSd);
  EXPECT_EQ(pooled.degrees_of_freedom, 9u);
  EXPECT_NEAR(pooled.standardizer, 3.033150177620620003, 1e-14);
  EXPECT_NEAR(pooled.cohen_d, 1.252822899451995653, 1e-14);
  ASSERT_TRUE(pooled.hedges_g);
  EXPECT_NEAR(*pooled.hedges_g, 1.144923391671924140, 1e-14);
  EXPECT_NEAR(pooled.confidence_interval.first, -.091587243522480155, 1e-9);
  EXPECT_NEAR(pooled.confidence_interval.second, 2.541784812838623342, 1e-9);

  const auto reverse = pooled_standardized_mean_difference(second, first);
  EXPECT_DOUBLE_EQ(reverse.cohen_d, -pooled.cohen_d);
  EXPECT_NEAR(reverse.confidence_interval.first, -pooled.confidence_interval.second, 1e-13);
  EXPECT_NEAR(reverse.confidence_interval.second, -pooled.confidence_interval.first, 1e-13);
}

TEST(StandardizedEffects, HandlesSmallSamplesAndExtremeCommonScales) {
  const auto two = one_sample_standardized_mean_difference({1, 3});
  EXPECT_EQ(two.degrees_of_freedom, 1u);
  EXPECT_FALSE(two.hedges_g);
  EXPECT_NEAR(two.cohen_d, std::sqrt(2.), 1e-14);
  EXPECT_TRUE(std::isfinite(two.confidence_interval.first));
  EXPECT_TRUE(std::isfinite(two.confidence_interval.second));

  const auto centered = one_sample_standardized_mean_difference({1e14 + 1, 1e14 + 3, 1e14 + 5}, 1e14);
  const auto unscaled = one_sample_standardized_mean_difference({1, 3, 5});
  EXPECT_DOUBLE_EQ(centered.cohen_d, unscaled.cohen_d);
  EXPECT_DOUBLE_EQ(centered.standardizer, 2);
  const auto scaled = one_sample_standardized_mean_difference({1e300, 3e300, 5e300});
  EXPECT_DOUBLE_EQ(scaled.cohen_d, unscaled.cohen_d);
  EXPECT_DOUBLE_EQ(scaled.standardizer, 2e300);

  const auto largest = std::numeric_limits<double>::max();
  const auto long_difference = paired_standardized_mean_difference({largest, largest, 0}, {-largest, 0, 0});
  EXPECT_DOUBLE_EQ(long_difference.cohen_d, 1);
  EXPECT_TRUE(std::isfinite(long_difference.confidence_interval.second));
  EXPECT_THROW(one_sample_standardized_mean_difference({1e300, 1e300, 1e300}), std::invalid_argument);
  EXPECT_THROW(paired_standardized_mean_difference({2, 3}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(pooled_standardized_mean_difference({2, 2}, {5, 5}), std::invalid_argument);
  EXPECT_THROW(pooled_standardized_mean_difference({1, 3}, {2, 3}, 1), std::invalid_argument);
}

} // namespace
