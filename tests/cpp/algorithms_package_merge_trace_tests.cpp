#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/package_merge.hpp>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
using Wide = unsigned __int128;
Wide cost(PackageMergeCost c) {
  return (Wide(c.high) << 64) | c.low;
}
Wide brute(const std::vector<std::uint64_t>& weights, int maximum) {
  Wide                                                best = ~Wide(0);
  std::size_t                                         full = std::size_t(1) << maximum;
  std::function<void(std::size_t, std::size_t, Wide)> search =
      [&](std::size_t i, std::size_t slots, Wide value) {
        if (i == weights.size()) {
          if (slots == full)
            best = std::min(best, value);
          return;
        }
        for (int length = 1; length <= maximum; ++length) {
          auto width = std::size_t(1) << (maximum - length);
          if (slots + width <= full)
            search(i + 1, slots + width, value + Wide(weights[i]) * length);
        }
      };
  search(0, 0, 0);
  return best;
}
} // namespace
TEST(PackageMergeTrace, ExhaustiveSmallPrefixLengthsIndependentOptimalCostAndEveryActualPackage) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 2; n <= 6; ++n)
    for (int maximum = 1; maximum <= 4; ++maximum) {
      if ((std::size_t(1) << maximum) < n)
        continue;
      for (int repeat = 0; repeat < 15; ++repeat) {
        std::vector<std::uint64_t> weights(n);
        for (auto& w : weights)
          w = 1 + rng() % 17;
        if (repeat == 0)
          weights[0] = UINT64_MAX;
        auto t = package_merge_trace(weights, maximum);
        EXPECT_EQ(cost(t.run.weighted_cost), brute(weights, maximum));
        EXPECT_TRUE(t.run.kraft_equality);
        EXPECT_LE(t.run.actual_max_length, maximum);
        EXPECT_EQ(package_merge(weights, maximum), t.run.lengths);
        std::size_t units = 2 + 2 * n, package_members = 0;
        for (std::size_t i = 0; i < t.coins.size(); ++i) {
          auto& coin = t.coins[i];
          EXPECT_EQ(coin.id, i);
          units += 1 + coin.members.size();
          if (coin.package) {
            ASSERT_LT(coin.left, i);
            ASSERT_LT(coin.right, i);
            auto members = t.coins[coin.left].members;
            members.insert(
                members.end(),
                t.coins[coin.right].members.begin(),
                t.coins[coin.right].members.end());
            EXPECT_EQ(coin.members, members);
            EXPECT_EQ(
                cost(coin.cost), cost(t.coins[coin.left].cost) + cost(t.coins[coin.right].cost));
            package_members += coin.members.size();
          } else {
            EXPECT_EQ(coin.members, (std::vector<std::size_t>{i}));
            EXPECT_EQ(cost(coin.cost), weights[i]);
          }
        }
        std::size_t pruned = 0;
        for (auto& level : t.levels) {
          units += 1 + level.coins.size();
          pruned += level.pruned;
          for (std::size_t i = 1; i < level.coins.size(); ++i)
            EXPECT_LE(cost(t.coins[level.coins[i - 1]].cost), cost(t.coins[level.coins[i]].cost));
          EXPECT_LE(level.coins.size(), 2 * n - 2);
        }
        units += t.selected.size();
        EXPECT_EQ(t.selected.size(), 2 * n - 2);
        EXPECT_EQ(
            t.run.work.member_copies, 2 * n + 2 * package_members + (t.levels.size() - 1) * n);
        EXPECT_EQ(t.run.work.pruned_coins, pruned);
        EXPECT_EQ(t.run.work.levels, t.levels.size());
        EXPECT_EQ(t.captured_entries, units);
        EXPECT_NO_THROW(package_merge_trace(weights, maximum, {0, 0, units}));
        EXPECT_THROW(package_merge_trace(weights, maximum, {0, 0, units - 1}), std::length_error);
        EXPECT_THROW(package_merge_run(weights, maximum, {n - 1, 0, 0}), std::length_error);
        EXPECT_THROW(
            package_merge_run(weights, maximum, {0, t.run.work.member_copies - 1, 0}),
            std::length_error);
      }
    }
}
TEST(PackageMergeTrace, EmptySingletonFeasibilityExactWideCostsAndOriginalSweep) {
  EXPECT_THROW(package_merge({}, 0), std::invalid_argument);
  EXPECT_THROW(package_merge({0}, 1), std::invalid_argument);
  EXPECT_THROW(package_merge({1, 1, 1}, 1), std::invalid_argument);
  EXPECT_TRUE(package_merge({}, 1).empty());
  auto single = package_merge_run({7}, 1);
  EXPECT_EQ(single.lengths, (std::vector<int>{1}));
  EXPECT_EQ(single.kraft_sum, .5L);
  EXPECT_FALSE(single.kraft_equality);
  auto wide = package_merge_run({UINT64_MAX, UINT64_MAX}, std::numeric_limits<int>::max());
  EXPECT_EQ(wide.effective_levels, 1);
  EXPECT_EQ(package_merge_cost_decimal(wide.weighted_cost), "36893488147419103230");
  EXPECT_EQ(wide.weighted_cost.high, 1);
  auto weights       = package_merge_demo_weights();
  auto unconstrained = package_merge_run(weights, int(weights.size()));
  Wide previous      = ~Wide(0);
  for (int maximum = 4; maximum <= 8; ++maximum) {
    auto r = package_merge_run(weights, maximum);
    EXPECT_LE(cost(r.weighted_cost), previous);
    EXPECT_GE(cost(r.weighted_cost), cost(unconstrained.weighted_cost));
    previous = cost(r.weighted_cost);
  }
  EXPECT_EQ(previous, cost(unconstrained.weighted_cost));
}
