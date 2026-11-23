#include <datamunge/stats/discovery_bounds.hpp>
#include <datamunge/stats/freedman_lane.hpp>
#include <datamunge/stats/p_adjust.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

TEST(MultipleTesting, StoreyThresholdsMatchScaledRAdjustmentsAndRetainTies) {
  const std::vector<double> p{.8, .001, .01, .01, .55, .3};
  const auto fits = storey_qvalues(p, {0, 1.0 / 3, .8}, false);
  ASSERT_EQ(fits.size(), 3u);
  EXPECT_EQ(fits[0].q_values, p_adjust(p, PAdjustMethod::BH));
  EXPECT_DOUBLE_EQ(fits[1].null_fraction, .5);
  const std::vector<double> expected{.4, .003, .01, .01, .33, .225};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(fits[1].q_values[i], expected[i], 1e-15);
  EXPECT_EQ(fits[2].q_values, std::vector<double>(p.size(), 0));
  EXPECT_NEAR(storey_qvalues(p, {.8}).front().null_fraction, 5.0 / 6, 1e-15);
  EXPECT_THROW(storey_qvalues(p, {1}), std::invalid_argument);
  EXPECT_THROW(storey_qvalues({std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}

TEST(MultipleTesting, SimesShortcutMatchesExhaustiveClosedTestingForEverySubset) {
  const std::vector<std::vector<double>> families{
      {.002, .015, .016, .055, .19, .7, .9},
      {.05, .08, .05, 0, .55, .01, .35},
      {.001, .001, .001, .001},
      {.4, .9, 1}
  };
  constexpr double alpha = .05;
  for (const auto& p : families) {
    const SimesDiscoveryBounds fit(p, alpha);
    const auto end = 1u << p.size();
    std::vector<unsigned> locally_nonrejected{0};
    for (unsigned mask = 1; mask < end; ++mask) {
      std::vector<double> ordered;
      for (std::size_t i = 0; i < p.size(); ++i)
        if ((mask >> i) & 1u)
          ordered.push_back(p[i]);
      std::sort(ordered.begin(), ordered.end());
      long double combined = 1;
      for (std::size_t i = 0; i < ordered.size(); ++i)
        combined = std::min(combined, ordered.size() * static_cast<long double>(ordered[i]) / (i + 1));
      if (combined > alpha)
        locally_nonrejected.push_back(mask);
    }
    for (unsigned selected = 0; selected < end; ++selected) {
      unsigned max_false = 0;
      for (unsigned subset = selected; subset; subset = (subset - 1) & selected) {
        const auto is_nonrejected = std::ranges::any_of(locally_nonrejected,
            [=](unsigned superset) { return (subset & superset) == subset; });
        if (is_nonrejected)
          max_false = std::max(max_false, static_cast<unsigned>(std::popcount(subset)));
      }
      std::vector<std::size_t> indices;
      for (std::size_t i = p.size(); i > 0; --i)
        if ((selected >> (i - 1)) & 1u)
          indices.push_back(i - 1);
      EXPECT_EQ(fit.lower_bound(indices), std::popcount(selected) - max_false);
    }
  }
  EXPECT_EQ(SimesDiscoveryBounds({.05}).lower_bound({0}), 1u);
  EXPECT_EQ(SimesDiscoveryBounds({}).lower_bound({}), 0u);
  EXPECT_THROW(SimesDiscoveryBounds({.5}).lower_bound({0, 0}), std::invalid_argument);
  EXPECT_THROW(SimesDiscoveryBounds({.5}).lower_bound({1}), std::invalid_argument);
}

class ScriptedPermutation : public datamunge::random::RandomGenerator {
public:
  std::uint64_t next_u64() override {
    return draws_++ == 5 ? (1ULL << 60) : std::numeric_limits<std::uint64_t>::max();
  }

private:
  std::size_t draws_{0};
};

TEST(ResidualPermutation, PartialFMatchesRAndKeepsTheNuisanceFit) {
  const auto nuisance = Matrix::from_columns({{1, 1, 1, 1, 1, 1}, {0, 0, 1, 1, 2, 2}});
  const Matrix tested(6, 1, {0, 1, 0, 1, 0, 1});
  ScriptedPermutation rng;
  const auto actual = freedman_lane_test({1, 5, 2, 7, 6, 4}, nuisance, tested, 2, rng);
  EXPECT_NEAR(actual.statistic, 1.67045454545454475, 1e-14);
  EXPECT_DOUBLE_EQ(actual.parameter1, 1);
  EXPECT_DOUBLE_EQ(actual.parameter2, 3);
  // Identity ties the observed statistic. Swapping rows one and six gives F = 0.70755 in R.
  EXPECT_DOUBLE_EQ(actual.p_value, 2.0 / 3);

  ScriptedPermutation replay;
  const auto rescaled = freedman_lane_test({1e200, 5e200, 2e200, 7e200, 6e200, 4e200},
      Matrix::from_columns({std::vector<double>(6, 1e100), {0, 0, 1, 1, 2, 2}}),
      tested, 2, replay);
  EXPECT_NEAR(rescaled.statistic, actual.statistic, 1e-14);
  EXPECT_EQ(rescaled.p_value, actual.p_value);
  EXPECT_THROW(freedman_lane_test({1, 5, 2, 7, 6, 4}, nuisance, nuisance, 1, rng), std::invalid_argument);

  const auto joint_nuisance = Matrix::from_columns({std::vector<double>(8, 1), {0, 0, 1, 1, 2, 2, 3, 3}});
  const auto joint_terms = Matrix::from_columns({{0, 1, 0, 1, 0, 1, 0, 1}, {0, 0, 1, 0, 1, 1, 0, 1}});
  const auto joint = freedman_lane_test({0, 2, 4, 1, 3, 5, 3, 7}, joint_nuisance, joint_terms, 1, rng);
  EXPECT_NEAR(joint.statistic, 5.4, 1e-14);
  EXPECT_DOUBLE_EQ(joint.parameter1, 2);
  EXPECT_DOUBLE_EQ(joint.parameter2, 4);
}

TEST(ResidualPermutation, SupportsAnEmptyReducedModelAndBoundaryStatistics) {
  datamunge::random::SplitMix64 rng(4);
  const auto fit = freedman_lane_test({3, 6, 9, 12}, Matrix(4, 0),
                                      Matrix(4, 1, {1, 2, 3, 4}), 29, rng);
  EXPECT_TRUE(std::isinf(fit.statistic));
  EXPECT_GT(fit.p_value, 0);
  EXPECT_LE(fit.p_value, 1);
  EXPECT_THROW(freedman_lane_test({2, 2, 2}, Matrix(3, 1, 1.0), Matrix(3, 1, {0, 1, 2}),
                                  29, rng), std::invalid_argument);
}

} // namespace
