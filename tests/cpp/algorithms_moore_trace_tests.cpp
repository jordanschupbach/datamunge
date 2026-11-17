#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/dfa_minimization.hpp>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MooreTrace, IndependentPairDistinguishabilityAndEveryActualSignature) {
  std::mt19937_64 rng(244253);
  for (int n = 1; n <= 9; ++n)
    for (int k = 0; k <= 3; ++k)
      for (int repeat = 0; repeat < 25; ++repeat) {
        DFA d;
        d.num_states  = n;
        d.num_symbols = k;
        d.start       = rng() % n;
        d.delta.assign(n, std::vector<int>(k));
        d.accept.resize(n);
        for (int i = 0; i < n; ++i) {
          d.accept[i] = rng() % 2 ? 2 : 0;
          for (auto& v : d.delta[i])
            v = rng() % n;
        }
        auto                           t = moore_minimize_trace(d);
        std::vector<std::vector<bool>> different(n, std::vector<bool>(n));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            different[i][j] = bool(d.accept[i]) != bool(d.accept[j]);
        bool changed = true;
        while (changed) {
          changed = false;
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
              if (!different[i][j])
                for (int c = 0; c < k; ++c)
                  if (different[d.delta[i][c]][d.delta[j][c]]) {
                    different[i][j] = true;
                    changed         = true;
                    break;
                  }
        }
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            EXPECT_EQ(t.run.partition[i] != t.run.partition[j], different[i][j]);
        EXPECT_TRUE(dfa_equivalence(d, t.run.result).equivalent);
        auto        previous = t.initial_partition;
        std::size_t units    = n;
        for (auto& r : t.rounds) {
          EXPECT_EQ(r.before, previous);
          std::vector<std::vector<int>> keys;
          std::vector<int>              after(n);
          for (int state = 0; state < n; ++state) {
            std::vector<int> key{previous[state]};
            for (auto dest : d.delta[state])
              key.push_back(previous[dest]);
            EXPECT_EQ(r.signatures[state], key);
            auto found   = std::find(keys.begin(), keys.end(), key);
            after[state] = int(found - keys.begin());
            if (found == keys.end())
              keys.push_back(key);
          }
          EXPECT_EQ(r.after, after);
          EXPECT_EQ(r.blocks_after, keys.size());
          EXPECT_EQ(r.stable, r.blocks_before == r.blocks_after);
          previous = after;
          units += 1 + 3 * n + n * (k + 1);
        }
        EXPECT_EQ(previous, t.run.partition);
        EXPECT_EQ(t.captured_entries, units);
        EXPECT_EQ(t.run.work.state_visits, n * t.rounds.size());
        EXPECT_EQ(t.run.work.transition_reads, n * k * t.rounds.size());
        EXPECT_GE(t.run.work.signature_components, t.run.work.map_comparisons);
        EXPECT_LE(t.rounds.size(), n);
        MooreOptions exact;
        exact.max_trace_entries = t.captured_entries;
        EXPECT_NO_THROW(moore_minimize_trace(d, exact));
        exact.max_trace_entries--;
        EXPECT_THROW(moore_minimize_trace(d, exact), std::length_error);
      }
}
TEST(MooreTrace, OriginalChainCompletionCapsAndReachability) {
  auto original = moore_minimize_trace(moore_demo_dfa());
  EXPECT_EQ(original.run.result.num_states, 3);
  EXPECT_EQ(original.run.work.rounds, 2);
  EXPECT_EQ(original.run.work.transition_reads, 24);
  EXPECT_EQ(original.captured_entries, 80);
  for (int n = 1; n <= 50; ++n) {
    auto r = moore_minimize_run(moore_chain_dfa(n));
    EXPECT_EQ(r.result.num_states, n);
    EXPECT_EQ(r.work.rounds, std::max(1, n - 1));
  }
  EXPECT_EQ(moore_minimize_trace(DFA{}).captured_entries, 6);
  DFA  partial{1, 1, {{-1}}, 0, {1}};
  auto p = moore_minimize_trace(partial);
  EXPECT_EQ(p.run.completed.num_states, 2);
  EXPECT_EQ(p.run.result.num_states, 2);
  EXPECT_TRUE(dfa_equivalence(partial, p.run.result).equivalent);
  MooreOptions o;
  o.max_states = 1;
  EXPECT_THROW(moore_minimize_trace(partial, o), std::length_error);
  o                 = {};
  o.max_transitions = 11;
  EXPECT_THROW(moore_minimize_trace(moore_demo_dfa(), o), std::length_error);
  o                   = {};
  o.max_trace_entries = 79;
  EXPECT_THROW(moore_minimize_trace(moore_demo_dfa(), o), std::length_error);
  auto broken        = moore_demo_dfa();
  broken.delta[5][0] = 100;
  EXPECT_THROW(moore_minimize_trace(broken, o), std::invalid_argument);
  DFA  no_start{2, 1, {{0}, {1}}, -1, {1, 0}};
  auto r = moore_minimize_run(no_start);
  EXPECT_EQ(r.completed.num_states, 3);
  EXPECT_EQ(r.result.num_states, 1);
  EXPECT_EQ(r.result.accept[0], 0);
  EXPECT_GT(r.blocks.size(), r.result.num_states);
}
