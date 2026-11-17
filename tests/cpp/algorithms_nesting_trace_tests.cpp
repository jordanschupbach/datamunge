#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/nesting.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NestingTrace, ExhaustiveDyadicFirstFitAndActualTrials) {
  for (int n = 0, total = 1; n <= 5; ++n, total *= 9)
    for (int code = 0; code < total; ++code) {
      int                 q = code, sum = 0;
      std::vector<double> sizes;
      std::vector<int>    units;
      for (int i = 0; i < n; ++i) {
        int value = q % 9;
        q /= 9;
        units.push_back(value);
        sizes.push_back(value / 8.);
        sum += value;
      }
      auto t = nesting_trace(sizes, 1);
      EXPECT_EQ(t.analysis.volume_lower_bound, (sum + 7) / 8);
      auto permutation = t.analysis.order;
      std::sort(permutation.begin(), permutation.end());
      for (int i = 0; i < n; ++i)
        EXPECT_EQ(permutation[i], i);
      for (int i = 1; i < n; ++i)
        EXPECT_GE(sizes[t.analysis.order[i - 1]], sizes[t.analysis.order[i]]);
      std::vector<int>              loads;
      std::vector<std::vector<int>> assignment;
      std::size_t                   trial = 0;
      for (std::size_t step = 0; step < t.analysis.order.size(); ++step) {
        int  item = t.analysis.order[step], chosen = -1;
        auto first = trial;
        for (std::size_t bin = 0; bin < loads.size(); ++bin) {
          auto& r = t.trials[trial++];
          EXPECT_EQ(r.item, item);
          EXPECT_EQ(r.bin, bin);
          EXPECT_EQ(r.load_before, loads[bin] / 8.);
          int compare = loads[bin] + units[item] < 8 ? -1 : loads[bin] + units[item] > 8 ? 1 : 0;
          EXPECT_EQ(r.comparison, compare);
          EXPECT_EQ(r.fits, compare <= 0);
          if (compare <= 0) {
            chosen = int(bin);
            break;
          }
        }
        bool opened = chosen < 0;
        if (opened) {
          chosen = int(loads.size());
          loads.push_back(0);
          assignment.emplace_back();
        }
        loads[chosen] += units[item];
        assignment[chosen].push_back(item);
        auto& p = t.placements[step];
        EXPECT_EQ(p.item, item);
        EXPECT_EQ(p.bin, chosen);
        EXPECT_EQ(p.opened, opened);
        EXPECT_EQ(p.first_trial, first);
        EXPECT_EQ(p.trial_count, trial - first);
        EXPECT_EQ(p.load_after, loads[chosen] / 8.);
      }
      EXPECT_EQ(t.analysis.result.assignment, assignment);
      EXPECT_EQ(t.analysis.result.bins, loads.size());
      EXPECT_EQ(t.analysis.bin_trials, trial);
      EXPECT_EQ(t.captured_entries, 1 + 4 * sizes.size() + loads.size() + trial);
      for (std::size_t i = 0; i < loads.size(); ++i)
        EXPECT_EQ(t.analysis.result.loads[i], loads[i] / 8.);
      EXPECT_TRUE(nesting_feasible(sizes, 1, assignment));
    }
}
TEST(NestingTrace, ExactBinaryCapacityVolumeRoundingAndCertificates) {
  const auto tiny = std::numeric_limits<double>::denorm_min(),
             big  = std::numeric_limits<double>::max();
  auto precision  = nesting_analyze({1, tiny}, 1);
  EXPECT_EQ(precision.result.bins, 2);
  EXPECT_EQ(precision.volume_lower_bound, 2);
  EXPECT_FALSE(nesting_feasible({1, tiny}, 1, {{0, 1}}));
  auto huge = nesting_analyze({big / 2, big / 2}, big);
  EXPECT_EQ(huge.result.bins, 1);
  EXPECT_EQ(huge.result.loads[0], big);
  EXPECT_EQ(huge.volume_lower_bound, 1);
  EXPECT_EQ(nesting_lower_bound({tiny, tiny}, tiny), 2);
  auto zeros = nesting_analyze({0, 0}, 1);
  EXPECT_EQ(zeros.result.bins, 1);
  EXPECT_EQ(zeros.volume_lower_bound, 0);
  EXPECT_EQ(nest_first_fit_decreasing({}, 1).bins, 0);
  std::vector<double> counter{.4, .4, .3, .3, .3, .3};
  EXPECT_EQ(nest_first_fit_decreasing(counter, 1).bins, 3);
  EXPECT_EQ(nesting_lower_bound(counter, 1), 2);
  EXPECT_TRUE(nesting_feasible(counter, 1, {{0, 2, 3}, {1, 4, 5}}));
  EXPECT_FALSE(nesting_feasible(counter, 1, {{0, 2, 3}, {0, 4, 5}}));
  EXPECT_FALSE(nesting_feasible(counter, 1, {{0}, {1}}));
  EXPECT_FALSE(nesting_feasible(counter, 1, {{-1}}));
  auto t = nesting_trace(nesting_demo_sizes(), 1);
  EXPECT_TRUE(nesting_feasible(t.sizes, 1, t.analysis.result.assignment));
  EXPECT_NO_THROW(nesting_trace(t.sizes, 1, t.captured_entries));
  EXPECT_THROW(nesting_trace(t.sizes, 1, t.captured_entries - 1), std::length_error);
  EXPECT_THROW(nest_first_fit_decreasing({}, 0), std::invalid_argument);
  EXPECT_THROW(nesting_lower_bound({2}, 1), std::invalid_argument);
  EXPECT_THROW(nesting_trace({-.1}, 1), std::invalid_argument);
  EXPECT_THROW(nesting_trace({std::nan("")}, 1), std::invalid_argument);
}
