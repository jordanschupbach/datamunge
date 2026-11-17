#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/grover_analysis.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
void certificate(int n, std::size_t target, int k) {
  auto       t = grover_trace(n, target, k);
  auto       r = grover_run(n, target, k);
  const auto p = grover_success_trace(n, target, k);
  const auto N = std::size_t{1} << n;
  EXPECT_EQ(t.run.state, r.state);
  EXPECT_EQ(t.run.result.measured, r.result.measured);
  ASSERT_EQ(t.steps.size(), k);
  ASSERT_EQ(p.size(), std::size_t(k) + 1);
  auto state = qstate_zero(n);
  for (int q = 0; q < n; ++q)
    h_gate(state, q);
  EXPECT_EQ(t.initial, state);
  EXPECT_EQ(p[0], std::norm(state[target]));
  for (int j = 0; j < k; ++j) {
    const auto& step = t.steps[j];
    EXPECT_EQ(step.iteration, j + 1);
    state[target] = -state[target];
    EXPECT_EQ(step.after_oracle, state);
    Complex mean{0, 0};
    for (auto z : state)
      mean += z;
    mean /= double(N);
    EXPECT_EQ(step.oracle_mean, mean);
    for (auto& z : state)
      z = 2. * mean - z;
    EXPECT_EQ(step.after_diffusion, state);
    EXPECT_EQ(step.marked_probability, std::norm(state[target]));
    EXPECT_EQ(p[j + 1], step.marked_probability);
    EXPECT_NEAR(step.marked_probability, grover_ideal_probability(n, j + 1), 2e-12);
  }
  EXPECT_EQ(state, r.state);
  EXPECT_EQ(r.result.success_prob, std::norm(state[target]));
  EXPECT_EQ(r.result.measured, most_probable(state));
  EXPECT_EQ(r.result.iterations, k);
  EXPECT_EQ(r.work.initialized_slots, N);
  EXPECT_EQ(r.work.hadamard_gates, n);
  EXPECT_EQ(r.work.hadamard_pairs, std::size_t(n) * (N / 2));
  EXPECT_EQ(r.work.oracle_calls, k);
  EXPECT_EQ(r.work.mean_terms, std::size_t(k) * N);
  EXPECT_EQ(r.work.diffusion_updates, std::size_t(k) * N);
  EXPECT_EQ(r.work.argmax_terms, N);
  EXPECT_EQ(t.captured_entries, N + std::size_t(k) * (2 * N + 1));
  EXPECT_NEAR(grover_state_mass(state), 1, 2e-12);
  GroverOptions o;
  o.max_amplitudes    = N;
  o.max_trace_entries = t.captured_entries;
  EXPECT_NO_THROW(grover_trace(n, target, k, o));
  if (t.captured_entries > 1) {
    --o.max_trace_entries;
    EXPECT_THROW(grover_trace(n, target, k, o), std::length_error);
  }
}
} // namespace
TEST(GroverTrace, ActualPhasesEveryTargetAndTheory) {
  for (int n = 0; n <= 6; ++n)
    for (std::size_t target = 0; target < (std::size_t{1} << n); ++target)
      certificate(n, target, 12);
}
TEST(GroverTrace, OriginalScheduleAndSmallStateCounterexamples) {
  EXPECT_EQ(grover(8, 42).iterations, 13);
  EXPECT_EQ(grover_first_peak_iterations(8), 12);
  EXPECT_GT(grover_run(8, 42, 12).result.success_prob, grover(8, 42).success_prob);
  EXPECT_EQ(grover(2, 3).iterations, 2);
  EXPECT_NEAR(grover(2, 3).success_prob, .25, 1e-14);
  EXPECT_EQ(grover(2, 3).measured, 0);
  EXPECT_GT(grover_run(2, 3, 1).result.success_prob, .999999999);
  EXPECT_EQ(grover(1, 1).measured, 0);
  EXPECT_NEAR(grover(1, 1).success_prob, .5, 1e-14);
  EXPECT_EQ(grover_first_peak_iterations(1), 0);
  EXPECT_EQ(grover_first_peak_iterations(0), 0);
  auto t                           = grover_trace(2, 3, 1);
  auto copy                        = t;
  copy.initial[0]                  = 99.;
  copy.steps[0].after_oracle[0]    = 99.;
  copy.steps[0].after_diffusion[0] = 99.;
  copy.run.state[0]                = 99.;
  EXPECT_NE(t.initial[0].real(), 99);
  EXPECT_NE(t.steps[0].after_oracle[0].real(), 99);
  EXPECT_NE(t.steps[0].after_diffusion[0].real(), 99);
  EXPECT_NE(t.run.state[0].real(), 99);
}
TEST(GroverTrace, CompleteCapsCheckedShiftsTargetsAndStandaloneStep) {
  EXPECT_THROW(grover(-1, 0), std::invalid_argument);
  EXPECT_THROW(grover(64, 0), std::invalid_argument);
  EXPECT_THROW(grover_run(2, 4, 1, {1, 1}), std::invalid_argument);
  EXPECT_THROW(grover_run(2, 3, -1), std::invalid_argument);
  EXPECT_THROW(grover_success_trace(2, 3, -1), std::invalid_argument);
  EXPECT_THROW(grover_run(3, 0, 0, {7, 0}), std::length_error);
  EXPECT_NO_THROW(grover_run(3, 0, 0, {8, 1}));
  EXPECT_THROW(grover_trace(2, 3, 1, {4, 12}), std::length_error);
  EXPECT_NO_THROW(grover_trace(2, 3, 1, {4, 13}));
  EXPECT_NO_THROW(grover_trace(2, 3, 1, {0, 0}));
  auto       state    = QuantumState{{1, 2}, {3, -1}, {-2, .5}};
  const auto before   = state;
  auto       expected = state;
  expected[1]         = -expected[1];
  Complex mean{0, 0};
  for (auto z : expected)
    mean += z;
  mean /= 3.;
  for (auto& z : expected)
    z = 2. * mean - z;
  detail::grover_step(state, 1);
  EXPECT_EQ(state, expected);
  auto invalid = before;
  EXPECT_THROW(detail::grover_step(invalid, 3), std::invalid_argument);
  EXPECT_EQ(invalid, before);
  QuantumState empty;
  EXPECT_THROW(detail::grover_step(empty, 0), std::invalid_argument);
  QuantumState huge{{1e308, 0}, {1e308, 0}, {1e308, 0}, {1e308, 0}};
  EXPECT_THROW(detail::grover_step(huge, 0), std::overflow_error);
}
