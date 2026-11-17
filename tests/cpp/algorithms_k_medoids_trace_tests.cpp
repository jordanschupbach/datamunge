#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/k_medoids.hpp>
#include <limits>
#include <random>
#include <set>
using namespace datamunge::algorithms;
namespace {
double objective(const std::vector<std::vector<double>>& d, const std::vector<std::size_t>& m) {
  double cost = 0;
  for (const auto& p : d) {
    double best = std::numeric_limits<double>::infinity();
    for (auto id : m)
      best = std::min(best, std::abs(p[0] - d[id][0]));
    cost += best;
  }
  return cost;
}
void verify(const KMedoidsTrace& t, std::size_t k) {
  const auto n = t.data.size();
  ASSERT_EQ(t.states.size(), t.result.iterations + 1);
  ASSERT_EQ(t.candidates.size(), t.work.candidates);
  EXPECT_EQ(t.captured_entries, n + t.states.size() + t.candidates.size());
  EXPECT_EQ(t.work.distance_pairs, n * (n - 1) / 2);
  EXPECT_EQ(t.work.assignment_passes, 1 + t.candidates.size() + t.result.iterations);
  EXPECT_EQ(t.work.distance_lookups, t.work.assignment_passes * n * k);
  for (std::size_t i = 0; i < t.states.size(); ++i) {
    const auto& s = t.states[i];
    EXPECT_EQ(s.medoids.size(), k);
    EXPECT_EQ(std::set<std::size_t>(s.medoids.begin(), s.medoids.end()).size(), k);
    EXPECT_EQ(s.assignment.size(), n);
    EXPECT_DOUBLE_EQ(s.cost, objective(t.data, s.medoids));
    for (std::size_t p = 0; p < n; ++p) {
      std::size_t want = 0;
      for (std::size_t c = 1; c < k; ++c)
        if (std::abs(t.data[p][0] - t.data[s.medoids[c]][0]) <
            std::abs(t.data[p][0] - t.data[s.medoids[want]][0]))
          want = c;
      EXPECT_EQ(s.assignment[p], want);
    }
    if (i)
      EXPECT_LT(s.cost, t.states[i - 1].cost - 1e-12);
  }
  for (const auto& e : t.candidates) {
    ASSERT_LT(e.iteration, t.states.size());
    auto m = t.states[e.iteration].medoids;
    EXPECT_EQ(std::find(m.begin(), m.end(), e.point), m.end());
    m[e.slot] = e.point;
    EXPECT_DOUBLE_EQ(e.cost, objective(t.data, m));
  }
  EXPECT_EQ(t.result.medoids, t.states.back().medoids);
  EXPECT_EQ(t.result.assignment, t.states.back().assignment);
  EXPECT_EQ(t.result.total_cost, t.states.back().cost);
  if (t.stop == KMedoidsStop::NoImprovement)
    for (std::size_t slot = 0; slot < k; ++slot)
      for (std::size_t o = 0; o < n; ++o)
        if (std::find(t.result.medoids.begin(), t.result.medoids.end(), o) ==
            t.result.medoids.end()) {
          auto m  = t.result.medoids;
          m[slot] = o;
          EXPECT_GE(objective(t.data, m), t.result.total_cost - 1e-12);
        }
}
} // namespace
TEST(KMedoidsTrace, IntegerLineObjectiveHistoryAndLocalCertificate) {
  std::mt19937 rng(184);
  for (int r = 0; r < 100; ++r) {
    std::vector<std::vector<double>> d;
    for (int i = 0, n = 1 + r % 9; i < n; ++i)
      d.push_back({double(int(rng() % 31) - 15)});
    for (std::size_t k = 1; k <= d.size(); ++k) {
      const auto t = k_medoids_trace(d, k, 30, r);
      verify(t, k);
      const auto plain = k_medoids(d, k, 30, r);
      EXPECT_EQ(plain.medoids, t.result.medoids);
      EXPECT_EQ(plain.assignment, t.result.assignment);
      EXPECT_EQ(plain.total_cost, t.result.total_cost);
    }
  }
}
TEST(KMedoidsTrace, ExactMedianAndInitializationLimit) {
  for (int seed = 0; seed < 20; ++seed) {
    const auto t = k_medoids_trace({{0}, {2}, {10}}, 1, 100, seed);
    EXPECT_EQ(t.result.medoids, std::vector<std::size_t>{1});
    EXPECT_EQ(t.result.total_cost, 10);
    EXPECT_EQ(t.stop, KMedoidsStop::NoImprovement);
    const auto init = k_medoids_trace({{0}, {2}, {10}}, 1, 0, seed);
    EXPECT_EQ(init.stop, KMedoidsStop::IterationLimit);
    EXPECT_TRUE(init.candidates.empty());
    EXPECT_EQ(init.states.size(), 1U);
    EXPECT_EQ(init.result.iterations, 0U);
  }
}
TEST(KMedoidsTrace, DuplicateTieIdentityAndZeroDimensionGuards) {
  const auto t = k_medoids_trace({{1}, {1}, {1}}, 2);
  verify(t, 2);
  EXPECT_EQ(t.result.assignment, std::vector<std::size_t>(3, 0));
  EXPECT_EQ(t.result.total_cost, 0);
  EXPECT_THROW((void)k_medoids({}, 1), std::invalid_argument);
  EXPECT_THROW((void)k_medoids({{}}, 1), std::invalid_argument);
  EXPECT_THROW((void)k_medoids({{0}, {1, 2}}, 1), std::invalid_argument);
  EXPECT_THROW((void)k_medoids({{0}}, 0), std::invalid_argument);
  EXPECT_THROW((void)k_medoids({{0}}, 2), std::invalid_argument);
  EXPECT_THROW(
      (void)k_medoids({{std::numeric_limits<double>::quiet_NaN()}}, 1), std::invalid_argument);
}
TEST(KMedoidsTrace, CompleteCapsAndOwnership) {
  const std::vector<std::vector<double>> d   = {{0}, {2}, {10}};
  const auto                             t   = k_medoids_trace(d, 1);
  const auto                             cap = t.captured_entries;
  EXPECT_NO_THROW((void)k_medoids_trace(d, 1, 100, 0, cap, 9));
  EXPECT_THROW((void)k_medoids_trace(d, 1, 100, 0, cap - 1, 9), std::length_error);
  EXPECT_THROW((void)k_medoids_trace(d, 1, 100, 0, 0, 8), std::length_error);
  EXPECT_NO_THROW((void)k_medoids_trace(d, 1, 100, 0, 0, 0));
  auto c                    = t;
  c.data[0][0]              = 99;
  c.states[0].assignment[0] = 99;
  c.result.medoids.clear();
  EXPECT_EQ(t.data[0][0], 0);
  EXPECT_LT(t.states[0].assignment[0], 1U);
  EXPECT_EQ(t.result.medoids.size(), 1U);
}
TEST(KMedoidsTrace, WidenedDistanceAndFiniteObjectiveControls) {
  const double tiny = std::numeric_limits<double>::denorm_min();
  const auto   t    = k_medoids_trace({{0}, {tiny}, {3 * tiny}}, 1);
  EXPECT_GT(t.result.total_cost, 0);
  EXPECT_NO_THROW((void)k_medoids({{0, 0}, {1e200, 1e200}}, 1));
  EXPECT_THROW(
      (void)k_medoids(
          {{std::numeric_limits<double>::max()}, {-std::numeric_limits<double>::max()}}, 1),
      std::overflow_error);
  EXPECT_THROW((void)k_medoids({{0,0}, {1e308,0}, {0,1e308}}, 1, 0, 1), std::overflow_error);
}
