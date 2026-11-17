#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/hopfield_analysis.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(HopfieldTrace, ExhaustiveThreeNeuronMemoriesAndProducingUpdates) {
  std::vector<std::vector<int>> all;
  for (unsigned mask = 0; mask < 8; ++mask) {
    std::vector<int> s(3);
    for (unsigned i = 0; i < 3; ++i)
      s[i] = (mask & (1u << i)) ? 1 : -1;
    all.push_back(s);
  }
  for (unsigned subset = 1; subset < 256; ++subset) {
    std::vector<std::vector<int>> patterns;
    for (unsigned p = 0; p < 8; ++p)
      if (subset & (1u << p))
        patterns.push_back(all[p]);
    const auto net = hopfield_train_checked(patterns);
    for (unsigned i = 0; i < 3; ++i)
      for (unsigned j = 0; j < 3; ++j) {
        int sum = 0;
        if (i != j)
          for (const auto& p : patterns)
            sum += p[i] * p[j];
        EXPECT_DOUBLE_EQ(net.weights[i][j], sum * (1. / 3));
      }
    for (const auto& initial : all)
      for (std::uint64_t seed : {0, 7}) {
        const auto t = hopfield_recall_trace(net, initial, 4, seed, {0, 0, 0});
        EXPECT_EQ(t.run.state, hopfield_recall(net, initial, 4, seed));
        auto        state   = initial;
        std::size_t updates = 0, flips = 0;
        double      last = t.initial_energy;
        for (const auto& sweep : t.sweeps) {
          auto order = sweep.order;
          std::sort(order.begin(), order.end());
          EXPECT_EQ(order, (std::vector<std::size_t>{0, 1, 2}));
          EXPECT_EQ(sweep.energy_before, last);
          std::size_t changed = 0;
          ASSERT_EQ(sweep.updates.size(), 3);
          for (std::size_t k = 0; k < 3; ++k) {
            const auto& u = sweep.updates[k];
            EXPECT_EQ(u.neuron, sweep.order[k]);
            EXPECT_EQ(u.before, state[u.neuron]);
            double field = 0;
            for (unsigned j = 0; j < 3; ++j)
              field += net.weights[u.neuron][j] * state[j];
            EXPECT_EQ(field, u.field);
            EXPECT_EQ(u.after, field >= 0 ? 1 : -1);
            state[u.neuron] = u.after;
            changed += u.before != u.after;
            ++updates;
          }
          EXPECT_EQ(state, sweep.state_after);
          EXPECT_EQ(changed, sweep.flips);
          flips += changed;
          double exact = 0;
          for (const auto& p : patterns) {
            int dot = 0;
            for (unsigned i = 0; i < 3; ++i)
              dot += p[i] * state[i];
            exact -= double(dot * dot - 3) / 6;
          }
          EXPECT_NEAR(sweep.energy_after, exact, 1e-12);
          EXPECT_LE(sweep.energy_after, sweep.energy_before + 1e-12);
          last = sweep.energy_after;
        }
        EXPECT_EQ(state, t.run.state);
        EXPECT_EQ(t.run.sweeps, t.sweeps.size());
        EXPECT_EQ(t.captured_entries, 4 + 10 * t.run.sweeps);
        EXPECT_EQ(t.run.work.field_terms, 9 * t.run.sweeps);
        EXPECT_EQ(t.run.work.neuron_updates, updates);
        EXPECT_EQ(t.run.work.flips, flips);
        EXPECT_EQ(t.run.work.diagnostic_energy_terms, 9 * (1 + t.run.sweeps));
        if (t.run.stop == HopfieldStop::NoChange) {
          ASSERT_FALSE(t.sweeps.empty());
          EXPECT_EQ(t.sweeps.back().flips, 0);
        }
      }
  }
}
TEST(HopfieldTrace, DomainsBeforeAllocationAndPolicies) {
  const auto net = hopfield_train({{1, -1}});
  EXPECT_THROW(hopfield_train_checked({{0, 1}}, {1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(hopfield_train({{INT_MAX, INT_MAX}}), std::invalid_argument);
  EXPECT_THROW(hopfield_train({{1}, {1, -1}}), std::invalid_argument);
  EXPECT_EQ(hopfield_train({{}, {}}).size(), 0);
  EXPECT_THROW(hopfield_energy(net, {1}), std::invalid_argument);
  EXPECT_THROW(hopfield_recall(net, {1, 0}), std::invalid_argument);
  EXPECT_THROW(hopfield_hamming({1}, {1, -1}), std::invalid_argument);
  EXPECT_THROW(hopfield_hamming({0}, {1}), std::invalid_argument);
  EXPECT_THROW(hopfield_energy(HopfieldNetwork{{{0}, {1, 0}}}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(
      hopfield_recall_run(HopfieldNetwork{{{0, 1}, {2, 0}}}, {1, 1}, 1, 0, {1, 1, 1}),
      std::invalid_argument);
  EXPECT_THROW(hopfield_energy(HopfieldNetwork{{{1}}}, {1}), std::invalid_argument);
  EXPECT_THROW(
      hopfield_energy(HopfieldNetwork{{{std::numeric_limits<double>::quiet_NaN()}}}, {1}),
      std::invalid_argument);
  EXPECT_THROW(
      hopfield_energy(
          HopfieldNetwork{
              {{0, std::numeric_limits<double>::infinity()},
               {std::numeric_limits<double>::infinity(), 0}}},
          {1, 1}),
      std::invalid_argument);
  EXPECT_THROW(hopfield_train_checked({{1, -1}}, {2, 3, 0}), std::length_error);
  EXPECT_THROW(hopfield_recall_run(net, {1, -1}, 1, 0, {1, 4, 0}), std::length_error);
  EXPECT_THROW(hopfield_detail::square(std::numeric_limits<std::size_t>::max()), std::length_error);
  const double          large = std::numeric_limits<double>::max();
  const HopfieldNetwork huge{{{0, large}, {large, 0}}};
  EXPECT_THROW(hopfield_energy(huge, {1, 1}), std::overflow_error);
  EXPECT_EQ(hopfield_recall_run(huge, {1, 1}, 1).stop, HopfieldStop::NoChange);
  EXPECT_THROW(hopfield_recall_trace(huge, {1, 1}, 0), std::overflow_error);
  const HopfieldNetwork field_overflow{{{0, large, large}, {large, 0, large}, {large, large, 0}}};
  EXPECT_THROW(hopfield_recall_run(field_overflow, {1, 1, 1}, 1), std::overflow_error);
}
TEST(HopfieldTrace, TiePlateauBudgetEmptyCaptureAndStoredCounterexample) {
  const auto one = hopfield_train({{-1}});
  const auto t   = hopfield_recall_trace(one, {-1}, 1, 0, {1, 1, 6});
  EXPECT_EQ(t.run.state, (std::vector<int>{1}));
  EXPECT_EQ(t.run.stop, HopfieldStop::SweepLimit);
  EXPECT_EQ(t.initial_energy, t.sweeps[0].energy_after);
  EXPECT_EQ(t.run.work.flips, 1);
  EXPECT_EQ(t.sweeps[0].updates[0].field, 0);
  EXPECT_THROW(hopfield_recall_trace(one, {-1}, 1, 0, HopfieldOptions{1, 1, 5}), std::length_error);
  EXPECT_EQ(hopfield_recall_run(one, {-1}, 1, 0, {1, 1, 1}).state, (std::vector<int>{1}));
  auto zero = hopfield_recall_trace(one, {-1}, 0);
  EXPECT_EQ(zero.run.sweeps, 0);
  EXPECT_EQ(zero.captured_entries, 2);
  EXPECT_EQ(zero.run.state, (std::vector<int>{-1}));
  auto empty = hopfield_recall_trace(HopfieldNetwork{}, {});
  EXPECT_EQ(empty.run.sweeps, 1);
  EXPECT_EQ(empty.captured_entries, 2);
  EXPECT_EQ(empty.run.stop, HopfieldStop::NoChange);
  auto copy                     = t;
  copy.sweeps[0].state_after[0] = -1;
  EXPECT_EQ(t.sweeps[0].state_after[0], 1);
  const auto net = hopfield_train({{1, 1, 1, 1}, {1, 1, 1, -1}, {1, 1, -1, -1}});
  const auto r   = hopfield_recall_run(net, {1, 1, 1, 1}, 20, 0);
  EXPECT_NE(r.state, (std::vector<int>{1, 1, 1, 1}));
  EXPECT_LT(hopfield_energy(net, r.state), hopfield_energy(net, {1, 1, 1, 1}));
}

TEST(HopfieldTrace, OriginalStudyAndNativeStateView) {
  const auto e = hopfield_original_experiment(1, 20240730, 11);
  EXPECT_EQ(e.network.size(), 100);
  EXPECT_EQ(e.memories.size(), 3);
  EXPECT_EQ(e.trials.size(), 11);
  for (std::size_t i = 0; i < e.trials.size(); ++i) {
    const auto& t = e.trials[i];
    const auto& s = e.summaries[i];
    EXPECT_EQ(t.initial_distance, hopfield_hamming(t.probe, e.memories[0]));
    EXPECT_EQ(t.final_distance, hopfield_hamming(t.recall.state, e.memories[0]));
    EXPECT_EQ(t.recall.state, hopfield_recall(e.network, t.probe, 100, t.recall_seed));
    EXPECT_EQ(s.exact_recoveries, t.final_distance == 0);
    EXPECT_DOUBLE_EQ(s.bit_agreement, double(100 - t.final_distance) / 100.);
    EXPECT_EQ(s.trials, 1);
  }
  EXPECT_THROW(hopfield_original_experiment(0), std::invalid_argument);
  EXPECT_THROW(hopfield_original_experiment(1, 0, 10), std::length_error);
  EXPECT_THROW(
      hopfield_original_experiment(std::numeric_limits<std::size_t>::max(), 0, 0),
      std::length_error);
  const auto p = hopfield_states_plot({{1, -1}, {-1, 1}}, "orientation");
  ASSERT_EQ(p.series().size(), 4);
  for (const auto& cell : p.series()) {
    const auto x = (*std::min_element(cell.x.begin(), cell.x.end()) +
                    *std::max_element(cell.x.begin(), cell.x.end())) /
                   2;
    const auto y = (*std::min_element(cell.y.begin(), cell.y.end()) +
                    *std::max_element(cell.y.begin(), cell.y.end())) /
                   2;
    const bool positive = y == 1 ? x == 0 : x == 1;
    EXPECT_EQ(int(cell.color.r), positive ? 37 : 51);
    EXPECT_EQ(int(cell.color.b), positive ? 235 : 85);
  }
  EXPECT_THROW(hopfield_states_plot({{1, 0}, {1, -1}}, "invalid", 1), std::invalid_argument);
  EXPECT_THROW(hopfield_states_plot({{1, -1}, {-1, 1}}, "cap", 3), std::length_error);
}
