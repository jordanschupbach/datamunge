#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/karger.hpp>
#include <limits>
#include <numeric>
#include <random>
using namespace datamunge::algorithms;
namespace {
std::size_t exact(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& e) {
  std::size_t best = e.size();
  for (std::size_t mask = 1; mask + 1 < (std::size_t(1) << n); ++mask) {
    std::size_t cut = 0;
    for (auto [u, v] : e)
      cut += ((mask >> u) & 1) != ((mask >> v) & 1);
    best = std::min(best, cut);
  }
  return best;
}
void check(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
  const auto t = karger_trace(n, edges, 16, 7);
  const auto r = karger_min_cut(n, edges, 16, 7);
  EXPECT_EQ(r.cut_size, t.result.cut_size);
  EXPECT_EQ(r.side_a, t.result.side_a);
  EXPECT_EQ(r.side_b, t.result.side_b);
  EXPECT_GE(r.cut_size, exact(n, edges));
  std::size_t     best = edges.size() + 1, steps = 0, draws = 0;
  std::mt19937_64 rng(7);
  for (std::size_t trial = 0; trial < t.trials.size(); ++trial) {
    std::vector<std::size_t> label(n);
    std::iota(label.begin(), label.end(), 0);
    const auto& end = t.trials[trial];
    EXPECT_EQ(end.first_step, steps);
    for (std::size_t k = 0; k < end.step_count; ++k) {
      const auto& s = t.steps[steps++];
      EXPECT_EQ(s.trial, trial);
      std::vector<std::size_t> crossing;
      for (std::size_t i = 0; i < edges.size(); ++i)
        if (label[edges[i].first] != label[edges[i].second])
          crossing.push_back(i);
      ASSERT_EQ(crossing.size(), s.crossing_count);
      ASSERT_LT(s.chosen_rank, crossing.size());
      EXPECT_EQ(s.edge_index, crossing[s.chosen_rank]);
      const auto bound     = std::uint64_t(crossing.size()),
                 threshold = (std::uint64_t(0) - bound) % bound;
      std::uint64_t word   = 0;
      for (std::size_t d = 0; d < s.draw_count; ++d) {
        word = rng();
        if (d + 1 < s.draw_count)
          EXPECT_LT(word, threshold);
        else
          EXPECT_GE(word, threshold);
      }
      draws += s.draw_count;
      EXPECT_EQ(word, s.accepted_word);
      EXPECT_EQ(s.chosen_rank, word % bound);
      const auto [u, v] = edges[s.edge_index];
      const auto a = label[u], b = label[v], minimum = std::min(a, b);
      ASSERT_NE(a, b);
      for (auto& x : label)
        if (x == a || x == b)
          x = minimum;
      EXPECT_EQ(label, s.labels);
      EXPECT_EQ(s.groups_after + 1, s.groups_before);
    }
    std::size_t              cut = 0;
    std::vector<std::size_t> a, b;
    for (std::size_t v = 0; v < n; ++v)
      (label[v] == label[0] ? a : b).push_back(v);
    for (auto [u, v] : edges)
      cut += label[u] != label[v];
    EXPECT_EQ(end.cut_size, cut);
    EXPECT_EQ(end.side_a, a);
    EXPECT_EQ(end.side_b, b);
    ASSERT_FALSE(a.empty());
    ASSERT_FALSE(b.empty());
    EXPECT_EQ(end.best_changed, cut < best);
    best = std::min(best, cut);
  }
  EXPECT_EQ(best, r.cut_size);
  EXPECT_EQ(t.work.trials, 16U);
  EXPECT_EQ(t.work.contractions, steps);
  EXPECT_EQ(t.work.random_draws, draws);
}
} // namespace
TEST(KargerTrace, EveryFourVertexSimpleGraphAndActualHistories) {
  for (unsigned mask = 0; mask < 64; ++mask) {
    std::vector<std::pair<std::size_t, std::size_t>> e;
    unsigned                                         k = 0;
    for (std::size_t u = 0; u < 4; ++u)
      for (std::size_t v = u + 1; v < 4; ++v, ++k)
        if (mask & (1U << k))
          e.emplace_back(u, v);
    check(4, e);
  }
}
TEST(KargerTrace, LoopsParallelDisconnectedAndBridge) {
  check(2, {{0, 0}, {0, 1}, {0, 1}, {1, 1}});
  check(5, {{0, 1}, {1, 2}, {0, 2}, {3, 4}});
  check(3, {});
  const auto r = karger_min_cut(4, {{0, 1}, {1, 2}, {2, 0}, {2, 3}}, 100, 7);
  EXPECT_EQ(r.cut_size, 1U);
}
TEST(KargerTrace, CompleteAndOnlineQuotaOwnershipGuards) {
  EXPECT_THROW((void)karger_trace(1, {}), std::invalid_argument);
  EXPECT_THROW((void)karger_trace(2, {{0, 2}}), std::invalid_argument);
  EXPECT_NO_THROW((void)karger_trace(2, {{0, 1}}, 1, 7, {3, 1, 3}));
  EXPECT_THROW((void)karger_trace(2, {{0, 1}}, 1, 7, {3, 1, 2}), std::length_error);
  EXPECT_THROW((void)karger_trace(2, {{0, 1}}, 2, 7, {3, 1, 100}), std::length_error);
  EXPECT_THROW((void)karger_trace(2, {{0, 1}}, 1, 7, {2, 1, 100}), std::length_error);
  EXPECT_THROW(
      (void)karger_min_cut(std::numeric_limits<std::size_t>::max() / 2, {}, 0), std::length_error);
  auto t     = karger_trace(3, {{0, 1}, {1, 2}, {2, 0}}, 1, 7, {0, 0, 0});
  auto copy  = t;
  t.edges[0] = {2, 2};
  EXPECT_NE(copy.edges[0], t.edges[0]);
}
