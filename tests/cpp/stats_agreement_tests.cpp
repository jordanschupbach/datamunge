#include <datamunge/stats/agreement.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace datamunge::stats;

const std::vector<double> first{1, 2, 4, 7, 8, 11, 15, 18};
const std::vector<double> second{1.4, 2.3, 3.7, 6.1, 8.9, 11.7, 15.3, 20.1};

TEST(MeasurementAgreement, BlandAltmanMatchesRLimitAndIntervalCalculations) {
  const auto fit = bland_altman(first, second);
  EXPECT_NEAR(fit.bias, -.4375, 1e-14);
  EXPECT_NEAR(fit.difference_sd, .879833588161502034, 1e-14);
  EXPECT_NEAR(fit.limits.first, -2.161942145185190345, 2e-8);
  EXPECT_NEAR(fit.limits.second, 1.286942145185189901, 2e-8);
  EXPECT_NEAR(fit.bias_interval.first, -1.173059287255554972, 1e-9);
  EXPECT_NEAR(fit.bias_interval.second, .298059287255554695, 1e-9);
  EXPECT_NEAR(fit.lower_interval.first, -3.476746774010999630, 2e-8);
  EXPECT_NEAR(fit.lower_interval.second, -.847137516359381060, 2e-8);
  EXPECT_NEAR(fit.upper_interval.first, -.027862483640619384, 2e-8);
  EXPECT_NEAR(fit.upper_interval.second, 2.601746774010999186, 2e-8);
  EXPECT_EQ(bland_altman(second, first).limits, (std::pair{-fit.limits.second, -fit.limits.first}));
  EXPECT_EQ(bland_altman({2, 3}, {1, 2}).lower_interval, (std::pair{1., 1.}));
  EXPECT_THROW(bland_altman({2}, {1}), std::invalid_argument);
}

TEST(MeasurementAgreement, LinConcordanceMatchesIndependentInfluenceCalculation) {
  const auto fit = lin_concordance(first, second);
  EXPECT_NEAR(fit.concordance, .9877984550561797, 1e-14);
  EXPECT_NEAR(fit.standard_error, .00502969976466754, 1e-14);
  EXPECT_NEAR(fit.interval.first, .972701519929042, 2e-8);
  EXPECT_NEAR(fit.interval.second, .9945692852786681, 2e-8);
  EXPECT_EQ(lin_concordance(second, first).interval, fit.interval);
  EXPECT_EQ(lin_concordance(first, first).interval, (std::pair{1.0, 1.0}));
  EXPECT_EQ(lin_concordance({-3, 0, 3}, {3, 0, -3}).interval, (std::pair{-1.0, -1.0}));

  auto scaled_first = first;
  auto scaled_second = second;
  for (auto& x : scaled_first)
    x *= 1e160;
  for (auto& y : scaled_second)
    y *= 1e160;
  EXPECT_NEAR(lin_concordance(scaled_first, scaled_second).standard_error, fit.standard_error, 1e-15);
  EXPECT_THROW(lin_concordance({1, 1, 1}, {1, 1, 1}), std::invalid_argument);
}

TEST(RaterAgreement, CohenMatchesOrdinalWeightsAndFleissUsesPooledMargins) {
  const std::vector<std::vector<std::size_t>> paired{{12, 3, 1}, {2, 8, 3}, {0, 4, 9}};
  EXPECT_NEAR(cohen_kappa(paired).kappa, .53571428571428570, 1e-15);
  EXPECT_NEAR(cohen_kappa(paired, KappaWeights::Linear).kappa, .62452107279693481, 1e-15);
  const auto quadratic = cohen_kappa(paired, KappaWeights::Quadratic);
  EXPECT_NEAR(quadratic.kappa, .71355498721227617, 1e-15);
  EXPECT_NEAR(quadratic.observed_agreement, .90476190476190477, 1e-15);
  EXPECT_NEAR(quadratic.expected_agreement, .66751700680272108, 1e-15);

  const auto fleiss = fleiss_kappa({{4, 0, 0}, {1, 3, 0}, {0, 2, 2}, {0, 0, 4}, {2, 1, 1}});
  EXPECT_NEAR(fleiss.kappa, .39849624060150374, 1e-15);
  EXPECT_NEAR(fleiss.observed_agreement, .6, 1e-15);
  EXPECT_NEAR(fleiss.expected_agreement, .335, 1e-15);

  constexpr auto count_limit = std::numeric_limits<std::size_t>::max();
  EXPECT_DOUBLE_EQ(cohen_kappa({{count_limit, 0}, {0, 1}}).kappa, 1);
  EXPECT_THROW(cohen_kappa({{3, 0}, {0, 0}}), std::invalid_argument);
  EXPECT_THROW(fleiss_kappa({{2, 0}, {3, 0}}), std::invalid_argument);
}

} // namespace
