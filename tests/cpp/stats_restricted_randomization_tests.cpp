#include <datamunge/stats/restricted_randomization.hpp>
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using datamunge::stats::RestrictedRandomization;

TEST(RestrictedRandomization, MatchedPairExactIntervalIncludesAttainedBreakpoints) {
  const RestrictedRandomization fit({4, 1, 3, 2}, {true, false, true, false}, {7, 7, 4, 4});
  const auto result = fit.test();
  EXPECT_DOUBLE_EQ(result.estimate1, 2);
  EXPECT_DOUBLE_EQ(result.statistic, 2);
  EXPECT_DOUBLE_EQ(result.parameter1, 4);
  EXPECT_DOUBLE_EQ(result.parameter2, 2);
  EXPECT_DOUBLE_EQ(result.p_value, .5);
  EXPECT_EQ(fit.confidence_interval(.5), (std::pair{1.0, 3.0}));
  for (const double effect : {1.0, 2.0, 3.0})
    EXPECT_DOUBLE_EQ(fit.test(effect).p_value, 1);
  EXPECT_DOUBLE_EQ(fit.test(std::nextafter(1.0, 0.0)).p_value, .5);
  EXPECT_DOUBLE_EQ(fit.test(std::nextafter(3.0, 4.0)).p_value, .5);
  constexpr auto inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(fit.confidence_interval(.95), (std::pair{-inf, inf}));
}

TEST(RestrictedRandomization, UnequalInterleavedBlocksMatchDirectSharpNullEnumeration) {
  const std::vector<double> y{2, 11, 3, 4, -1, 5, 9};
  const std::vector<bool> treated{true, true, false, false, false, true, false};
  const std::vector<std::size_t> blocks{7, 8, 7, 8, 7, 8, 8};
  const RestrictedRandomization fit(y, treated, blocks, 18);
  const auto contrast = [&](unsigned assignment, long double null_effect) {
    long double group_a = 0, group_b = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      const auto adjusted = y[i] - (treated[i] ? null_effect : 0);
      const bool is_treated = (assignment >> i) & 1u;
      if (blocks[i] == 7)
        group_a += is_treated ? adjusted : -adjusted / 2;
      else
        group_b += is_treated ? adjusted / 2 : -adjusted / 2;
    }
    return (3 * group_a + 4 * group_b) / 7;
  };
  const unsigned mask_a = 1u | (1u << 2) | (1u << 4);
  const unsigned mask_b = 127u ^ mask_a;
  const unsigned observed = 1u | (1u << 1) | (1u << 5);
  const auto interval = fit.confidence_interval(.8);
  ASSERT_TRUE(std::isfinite(interval.first) && std::isfinite(interval.second));
  for (int quarter = -60; quarter <= 60; ++quarter) {
    const double effect = quarter / 4.0;
    const auto observed_abs = std::abs(contrast(observed, effect));
    int extreme = 0;
    for (unsigned assignment = 0; assignment < 128; ++assignment)
      if (std::popcount(assignment & mask_a) == 1 && std::popcount(assignment & mask_b) == 2)
        extreme += std::abs(contrast(assignment, effect)) + 1e-15L >= observed_abs;
    EXPECT_EQ(fit.test(effect).parameter2, extreme);
    EXPECT_EQ(effect >= interval.first && effect <= interval.second, extreme / 18.0 > .2);
  }
  EXPECT_GT(fit.test(interval.first).p_value, .2);
  EXPECT_GT(fit.test(interval.second).p_value, .2);
  EXPECT_THROW((RestrictedRandomization{y, treated, blocks, 17}), std::length_error);
}

class FirstChoice : public datamunge::random::RandomGenerator {
public:
  std::uint64_t next_u64() override { return 1ULL << 60; }
};

TEST(RestrictedRandomization, SimulationUsesObservedAssignmentCorrectionAndCanBeInverted) {
  FirstChoice rng;
  const RestrictedRandomization monte({4, 1, 3, 2}, {true, false, false, true}, {0, 0, 1, 1}, 9, rng);
  EXPECT_DOUBLE_EQ(monte.test(-4).parameter2, 0);
  EXPECT_DOUBLE_EQ(monte.test(-4).p_value, .1);
  EXPECT_DOUBLE_EQ(monte.test(1).p_value, 1);
  EXPECT_EQ(monte.confidence_interval(.8), (std::pair{-1.0, 3.0}));
  EXPECT_THROW((RestrictedRandomization{{4, 1}, {true, true}, {9, 9}}), std::invalid_argument);
}

} // namespace
