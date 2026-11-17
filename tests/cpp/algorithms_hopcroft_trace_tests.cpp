#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/dfa_minimization_analysis.hpp>
#include <numeric>
#include <set>
using namespace datamunge::algorithms;
TEST(HopcroftTrace, ExhaustiveTwoStatePartialMachines) {
  for (unsigned table = 0; table < 81; ++table)
    for (unsigned accept = 0; accept < 4; ++accept)
      for (int start = -1; start < 2; ++start) {
        unsigned code = table;
        DFA      d{2, 2, {{0, 0}, {0, 0}}, start, {char(accept & 1), char((accept >> 1) & 1)}};
        for (auto& row : d.delta)
          for (auto& target : row) {
            target = int(code % 3) - 1;
            code /= 3;
          }
        const auto  t    = hopcroft_minimize_trace(d, {0, 0, 0});
        const auto  full = complete_dfa(d);
        const auto& r    = t.run;
        EXPECT_TRUE(dfa_equivalence(d, r.result).equivalent);
        EXPECT_EQ(moore_minimize(d).num_states, r.result.num_states);
        for (int u = 0; u < r.result.num_states; ++u)
          for (int v = 0; v < u; ++v) {
            auto left = r.result, right = r.result;
            left.start  = u;
            right.start = v;
            EXPECT_FALSE(dfa_equivalence(left, right).equivalent);
          }
        EXPECT_EQ(r.completed_states, full.num_states);
        std::vector<std::set<int>> blocks;
        std::size_t                units = 0;
        for (const auto& b : t.initial_blocks) {
          blocks.emplace_back(b.begin(), b.end());
          units += 1 + b.size();
        }
        std::size_t splits = 0, states = 0, visits = 0, targets = 0;
        for (const auto& p : t.passes) {
          std::set<int> pred;
          for (int s = 0; s < full.num_states; ++s)
            if (std::find(p.splitter.begin(), p.splitter.end(), full.delta[s][p.symbol]) !=
                p.splitter.end())
              pred.insert(s);
          EXPECT_EQ(std::vector<int>(pred.begin(), pred.end()), p.predecessors);
          units += 1 + p.splitter.size() + p.predecessors.size();
          visits += pred.size();
          targets += p.splitter.size();
          if (!pred.empty())
            states += full.num_states;
          for (const auto& split : p.splits) {
            auto at = std::find(
                blocks.begin(),
                blocks.end(),
                std::set<int>(split.parent.begin(), split.parent.end()));
            ASSERT_NE(at, blocks.end());
            std::vector<int> inside, outside;
            for (int s : *at)
              (pred.count(s) ? inside : outside).push_back(s);
            EXPECT_EQ(inside, split.inside);
            EXPECT_EQ(outside, split.outside);
            ASSERT_FALSE(inside.empty());
            ASSERT_FALSE(outside.empty());
            const auto position = at - blocks.begin();
            blocks.erase(at);
            blocks.insert(blocks.begin() + position, std::set<int>(outside.begin(), outside.end()));
            blocks.insert(blocks.begin() + position, std::set<int>(inside.begin(), inside.end()));
            if (split.schedule == HopcroftSchedule::Inside)
              EXPECT_LE(inside.size(), outside.size());
            if (split.schedule == HopcroftSchedule::Outside)
              EXPECT_LT(outside.size(), inside.size());
            units += 1 + split.parent.size() + inside.size() + outside.size();
            ++splits;
          }
        }
        for (std::size_t b = 0; b < blocks.size(); ++b)
          for (int s : blocks[b])
            EXPECT_EQ(r.blocks[s], b);
        EXPECT_EQ(t.captured_entries, units);
        EXPECT_EQ(r.work.partition_states, states);
        EXPECT_EQ(r.work.predecessor_edges, visits);
        EXPECT_EQ(r.work.predecessor_targets, targets);
        EXPECT_EQ(r.work.splits, splits);
        EXPECT_EQ(r.work.symbol_passes, t.passes.size());
        EXPECT_EQ(r.work.prepared_transitions, full.num_states * full.num_symbols);
        EXPECT_LE(r.work.splitter_pops, r.completed_states);
      }
}
TEST(HopcroftTrace, PoliciesAndDegeneracy) {
  const DFA  d{3, 1, {{1}, {2}, {2}}, 0, {0, 0, 1}};
  const auto t = hopcroft_minimize_trace(d);
  EXPECT_EQ(t.run.result.num_states, 3);
  EXPECT_EQ(
      hopcroft_minimize_trace(d, {3, 3, t.captured_entries}).captured_entries, t.captured_entries);
  EXPECT_THROW(
      hopcroft_minimize_trace(d, HopcroftOptions{3, 3, t.captured_entries - 1}), std::length_error);
  EXPECT_THROW(hopcroft_minimize_trace(d, HopcroftOptions{2, 3, 0}), std::length_error);
  EXPECT_THROW(hopcroft_minimize_trace(d, HopcroftOptions{3, 2, 0}), std::length_error);
  EXPECT_EQ(hopcroft_minimize_run(d, {3, 3, 1}).result.num_states, 3);
  EXPECT_THROW(hopcroft_minimize_run(DFA{2, 1, {{2}, {0}}, 0, {0, 0}}, {1, 1, 1}), std::invalid_argument);
  EXPECT_EQ(hopcroft_minimize_run(DFA{}).completed_states, 1);
  EXPECT_EQ(hopcroft_minimize_trace(DFA{}).captured_entries, 2);
  EXPECT_THROW(hopcroft_minimize_trace(DFA{}, HopcroftOptions{0, 0, 1}), std::length_error);
  const DFA  zero{2, 0, {{}, {}}, 1, {0, 1}};
  const auto z = hopcroft_minimize_trace(zero);
  EXPECT_TRUE(z.passes.empty());
  EXPECT_EQ(z.run.work.splitter_pops, 2);
  EXPECT_EQ(z.run.result.num_states, 1);
  auto copy                 = t;
  copy.initial_blocks[0][0] = 42;
  EXPECT_NE(t.initial_blocks[0][0], 42);
}

TEST(HopcroftTrace, ReusableFixturesAndVirtualLimits) {
  const auto original = hopcroft_example_dfa();
  EXPECT_EQ(original.num_states, 6);
  EXPECT_EQ(hopcroft_minimize(original).num_states, 3);
  for (std::size_t threshold = 0; threshold < 12; ++threshold) {
    const auto d = dfa_unary_threshold(threshold, threshold + 1);
    EXPECT_EQ(d.num_states, threshold + 1);
    EXPECT_EQ(hopcroft_minimize(d).num_states, d.num_states);
    for (std::size_t length = 0; length < 14; ++length)
      EXPECT_EQ(dfa_accepts(d, std::vector<int>(length, 0)), length >= threshold);
  }
  EXPECT_THROW(dfa_unary_threshold(3, 3), std::length_error);
  EXPECT_THROW(dfa_unary_threshold(std::numeric_limits<std::size_t>::max()), std::length_error);
  EXPECT_THROW(
      dfa_unary_threshold(static_cast<std::size_t>(std::numeric_limits<int>::max()), 0),
      std::length_error);
  EXPECT_EQ(dfa_unary_threshold(3, 0).num_states, 4);
}
