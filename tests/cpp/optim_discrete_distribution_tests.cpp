#include <datamunge/optim/discrete_distribution_search.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace o = datamunge::optim;

TEST(DiscreteDistributionSearch, EveryMethodReplaysPartialBatchesWithFeasiblePoints) {
  for (const auto method : {o::DiscreteDistributionMethod::cross_entropy,
                            o::DiscreteDistributionMethod::umda, o::DiscreteDistributionMethod::pbil,
                            o::DiscreteDistributionMethod::compact_ga, o::DiscreteDistributionMethod::mimic,
                            o::DiscreteDistributionMethod::boa}) {
    for (const auto budget : {1u, 73u}) {
      std::vector<o::CategoricalPoint> first, second;
      const auto value = [](const o::CategoricalPoint& x) {
        return static_cast<double>((x[0] != x[1]) + (x[2] != x[3]) + (x[1] != x[3]));
      };
      const auto f = [&](const o::CategoricalPoint& x) { first.push_back(x); return value(x); };
      const auto replay = [&](const o::CategoricalPoint& x) { second.push_back(x); return value(x); };
      o::DiscreteDistributionOptions options{budget, 7, method};
      options.population = 8;
      const auto result = o::discrete_distribution_search(f, {2, 2, 2, 2}, options);
      EXPECT_EQ(result.evaluations, budget);
      EXPECT_EQ(first.size(), budget);
      EXPECT_EQ(result.value, value(result.point));
      for (const auto& point : first) {
        ASSERT_EQ(point.size(), 4);
        EXPECT_TRUE(std::ranges::all_of(point, [](auto v) { return v < 2; }));
        EXPECT_LE(result.value, value(point));
      }
      EXPECT_EQ(o::discrete_distribution_search(replay, {2, 2, 2, 2}, options).point, result.point);
      EXPECT_EQ(first, second);
    }
  }
}

TEST(DiscreteDistributionSearch, CrossEntropySupportsDifferentCardinalitiesAndFixedAxes) {
  const auto f = [](const o::CategoricalPoint& x) {
    EXPECT_EQ(x[0], 0);
    EXPECT_LT(x[1], 3);
    EXPECT_LT(x[2], 5);
    return 42;
  };
  const auto result = o::discrete_distribution_search(f, {1, 3, 5}, {11});
  EXPECT_EQ(result.evaluations, 11);
  EXPECT_EQ(result.value, 42);
  EXPECT_THROW(o::discrete_distribution_search(f, {1, 3, 5},
      {11, 7, o::DiscreteDistributionMethod::pbil}), std::invalid_argument);
  EXPECT_THROW(o::discrete_distribution_search(f, {0}), std::invalid_argument);
}
