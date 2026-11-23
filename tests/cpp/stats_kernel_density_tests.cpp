#include <datamunge/stats/kernel_density.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace datamunge::stats;

const std::vector<double> sample{.01, .04, .09, .18, .25, .31, .48, .57, .81, 1.08, 1.47, 2.13, 2.61, 3.58};
constexpr auto inf = std::numeric_limits<double>::infinity();

TEST(KernelDensity, GaussianIntervalMassRetainsSmallTailsAndWidths) {
  const GaussianKernelDensity kde({0}, 1);
  EXPECT_NEAR(kde.density(0), .3989422804014327, 1e-16);
  EXPECT_NEAR(kde.probability(-1, 1), .6826894921370859, 1e-15);
  EXPECT_NEAR(kde.probability(8, 9) / 6.2198319858658314e-16, 1, 1e-14);
  EXPECT_DOUBLE_EQ(kde.probability(8, 9), kde.probability(-9, -8));
  EXPECT_NEAR(kde.probability(0, 1e-20) / 3.989422804014327e-21, 1, 1e-15);
  EXPECT_DOUBLE_EQ(kde.probability(-inf, inf), 1);
  EXPECT_DOUBLE_EQ(kde.probability(1, 1), 0);
  EXPECT_THROW(kde.probability(2, 1), std::invalid_argument);
  EXPECT_THROW(kde.density(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST(KernelDensity, ReflectionNormalizesOnEitherHalfLine) {
  const auto h = .38706163514343295;
  const GaussianKernelDensity lower(sample, h, KdeBoundary{0});
  EXPECT_NEAR(lower.density(0), .93310907741490656, 1e-15);
  EXPECT_NEAR(lower.probability(0, .5), .41154913204411886, 1e-15);
  EXPECT_DOUBLE_EQ(lower.probability(-inf, -.1), 0);
  EXPECT_DOUBLE_EQ(lower.probability(-inf, inf), 1);
  EXPECT_DOUBLE_EQ(lower.density(-.1), 0);
  EXPECT_NEAR(lower.probability(.5, inf) + lower.probability(0, .5), 1, 1e-15);
  auto negative = sample;
  for (auto& v : negative)
    v = -v;
  const GaussianKernelDensity upper(negative, h, KdeBoundary{0, false});
  EXPECT_DOUBLE_EQ(upper.density(-.5), lower.density(.5));
  EXPECT_DOUBLE_EQ(upper.probability(-.5, 1), lower.probability(-1, .5));
  EXPECT_THROW((GaussianKernelDensity{sample, h, KdeBoundary{1}}), std::invalid_argument);
}

TEST(KernelDensity, BandwidthsMatchUnbinnedRAndScaleWithObservations) {
  const auto sj = kde_sheather_jones_bandwidth(sample);
  const auto cv = kde_lscv_bandwidth(sample, std::pair{.05, 1.5});
  EXPECT_NEAR(sj, .38706163514343295, 1e-14);
  EXPECT_NEAR(cv.bandwidth, .35608272517892969, 1e-11);
  EXPECT_NEAR(cv.objective, -.32635604867113421, 1e-14);
  EXPECT_FALSE(cv.at_boundary);
  auto transformed = sample;
  for (auto& v : transformed)
    v = 3 * v - 5;
  EXPECT_NEAR(kde_sheather_jones_bandwidth(transformed), 3 * sj, 1e-14);
  const auto scaled_cv = kde_lscv_bandwidth(transformed, std::pair{.15, 4.5});
  EXPECT_NEAR(scaled_cv.bandwidth, 3 * cv.bandwidth, 1e-11);
  EXPECT_NEAR(scaled_cv.objective, cv.objective / 3, 1e-14);
  const auto restricted = kde_lscv_bandwidth(sample, std::pair{.5, .9});
  EXPECT_DOUBLE_EQ(restricted.bandwidth, .5);
  EXPECT_TRUE(restricted.at_boundary);
  EXPECT_THROW(kde_sheather_jones_bandwidth({1, 1}), std::invalid_argument);
  EXPECT_THROW(kde_lscv_bandwidth({1, 1}), std::invalid_argument);
  EXPECT_THROW(kde_lscv_bandwidth(std::vector<double>(11, .1)), std::invalid_argument);
  EXPECT_DOUBLE_EQ(kde_lscv_bandwidth({1, 1}, std::pair{.1, 1.0}).bandwidth, .1);
}

} // namespace
