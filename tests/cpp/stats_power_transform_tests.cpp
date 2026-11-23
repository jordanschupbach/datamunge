#include <datamunge/stats/power_transform.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using namespace datamunge::stats;

TEST(PowerTransforms, MatchesIndependentlyProfiledGaussianLikelihoods) {
  const std::vector<double> positive{1.3, 1.6, 1.8, 2.7, 4.5, 7.1, 19.5, 45.2};
  const auto box = fit_box_cox(positive);
  EXPECT_EQ(box.family, PowerTransformFamily::BoxCox);
  EXPECT_NEAR(box.lambda, -.40719529375964197, 1e-6);
  EXPECT_NEAR(box.profile_log_likelihood, -13.30852786015972455, 1e-12);
  EXPECT_FALSE(box.at_search_limit);
  for (const auto x : positive)
    EXPECT_NEAR(box.inverse(box.apply(x)), x, 1e-12);

  const std::vector<double> signed_sample{-12, -7.5, -3, -1.1, 0, .2, .5, 3, 6, 20};
  const auto yj = fit_yeo_johnson(signed_sample);
  EXPECT_EQ(yj.family, PowerTransformFamily::YeoJohnson);
  EXPECT_NEAR(yj.lambda, .8677822943163197, 1e-6);
  EXPECT_NEAR(yj.profile_log_likelihood, -20.3493797456169929, 1e-12);
  EXPECT_FALSE(yj.at_search_limit);
  for (const auto x : signed_sample)
    EXPECT_NEAR(yj.inverse(yj.apply(x)), x, 1e-12);

  const auto restricted = fit_box_cox(positive, 0, 5);
  EXPECT_DOUBLE_EQ(restricted.lambda, 0);
  EXPECT_TRUE(restricted.at_search_limit);
  EXPECT_NEAR(restricted.profile_log_likelihood, -13.960830565708049, 1e-13);
  const auto opposite_limit = fit_yeo_johnson(signed_sample, -5, 0);
  EXPECT_DOUBLE_EQ(opposite_limit.lambda, 0);
  EXPECT_TRUE(opposite_limit.at_search_limit);
}

TEST(PowerTransforms, HandlesLogarithmicLimitsAndRescalesWithoutOverflow) {
  for (double lambda : {0.0, 1e-16, -1e-16}) {
    const FittedPowerTransform box{PowerTransformFamily::BoxCox, lambda, 0, false};
    EXPECT_NEAR(box.apply(.25), std::log(.25), 1e-15);
    EXPECT_NEAR(box.inverse(box.apply(.25)), .25, 1e-15);
  }
  const FittedPowerTransform zero{PowerTransformFamily::YeoJohnson, 0, 0, false};
  const FittedPowerTransform two{PowerTransformFamily::YeoJohnson, 2, 0, false};
  EXPECT_DOUBLE_EQ(zero.apply(4), std::log1p(4));
  EXPECT_DOUBLE_EQ(two.apply(-4), -std::log1p(4));
  EXPECT_DOUBLE_EQ(zero.apply(-4), -12);
  EXPECT_DOUBLE_EQ(two.apply(4), 12);
  EXPECT_DOUBLE_EQ(zero.inverse(zero.apply(-4)), -4);
  EXPECT_DOUBLE_EQ(two.inverse(two.apply(4)), 4);
  const FittedPowerTransform large_power{PowerTransformFamily::BoxCox, 1200, 0, false};
  const auto x = std::exp(713. / 1200);
  const auto transformed = large_power.apply(x);
  EXPECT_TRUE(std::isfinite(transformed));
  EXPECT_DOUBLE_EQ(large_power.inverse(transformed), x);

  const auto unscaled = fit_box_cox({1, 2, 3, 10, 30});
  const auto large = fit_box_cox({1e250, 2e250, 3e250, 1e251, 3e251});
  EXPECT_NEAR(large.lambda, unscaled.lambda, 1e-5);
  EXPECT_NEAR(large.profile_log_likelihood, unscaled.profile_log_likelihood - 5 * std::log(1e250), 1e-10);
  const auto tiny = fit_yeo_johnson({-2e-300, 0, 2e-300});
  EXPECT_TRUE(std::isfinite(tiny.profile_log_likelihood));
  EXPECT_NEAR(tiny.profile_log_likelihood, -3 * (std::log(2e-300) + .5 * std::log(2. / 3)), 1e-10);
  const auto wide = fit_yeo_johnson({-1e300, 0, 1e300});
  EXPECT_NEAR(wide.lambda, 1, 1e-6);
  EXPECT_TRUE(std::isfinite(wide.profile_log_likelihood));
}

TEST(PowerTransforms, ValidatesInputsAndFiniteInverseEndpoints) {
  const FittedPowerTransform positive{PowerTransformFamily::BoxCox, 2, 0, false};
  const FittedPowerTransform negative{PowerTransformFamily::YeoJohnson, 4, 0, false};
  EXPECT_THROW(positive.apply(0), std::invalid_argument);
  EXPECT_THROW(positive.inverse(-.5), std::invalid_argument);
  EXPECT_THROW(negative.inverse(-.5), std::invalid_argument);
  EXPECT_THROW(positive.apply(1e-100), std::overflow_error);
  EXPECT_THROW(negative.apply(-1e100), std::overflow_error);
  EXPECT_NEAR(negative.inverse(negative.apply(-4)), -4, 1e-14);
  EXPECT_THROW(fit_box_cox({0, 1}), std::invalid_argument);
  EXPECT_THROW(fit_box_cox({1, 1}), std::invalid_argument);
  EXPECT_THROW(fit_yeo_johnson({2, 9}, 2, -1), std::invalid_argument);
  EXPECT_THROW(fit_yeo_johnson({1, std::numeric_limits<double>::infinity()}), std::invalid_argument);
}

} // namespace
