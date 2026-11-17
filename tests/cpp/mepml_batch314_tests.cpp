#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/division.hpp>
#include <datamunge/algorithms/random_restart_hill_climbing.hpp>
#include <datamunge/algorithms/range_coding.hpp>
#include <datamunge/algorithms/ransac.hpp>
#include <datamunge/algorithms/rayleigh_quotient_iteration.hpp>
#include <datamunge/algorithms/rbf_network.hpp>
#include <datamunge/algorithms/rc4.hpp>
#include <datamunge/algorithms/realtime_scheduling.hpp>
#include <datamunge/algorithms/recursive_descent_parser.hpp>
#include <datamunge/algorithms/reference_counting.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/rbf.hpp>
#include <numeric>
#include <random>
namespace a = datamunge::algorithms;
namespace n = datamunge::nn;
TEST(Batch314Hill, SharedProducerAndActualObjective) {
  a::HillClimbOptions o;
  o.restarts = 3;
  o.seed     = 7;
  auto f     = [](const auto& x) { return -(x[0] - 2) * (x[0] - 2); };
  auto t     = a::hill_climb_trace(f, {0}, {4}, o);
  auto r     = a::hill_climb_run(f, {0}, {4}, o);
  EXPECT_EQ(t.best.x, r.best.x);
  EXPECT_NEAR(t.best.x[0], 2, 1e-7);
  EXPECT_EQ(t.evaluations, r.evaluations);
  std::size_t eval = 0;
  for (const auto& s : t.steps)
    if (s.evaluated) {
      ++eval;
      EXPECT_DOUBLE_EQ(s.candidate_value, f(s.candidate));
    }
  EXPECT_EQ(eval, t.evaluations);
  o.restarts = 0;
  EXPECT_FALSE(a::hill_climb_run(f, {0}, {4}, o).found);
}
TEST(Batch314Hill, ValidationAndBudgets) {
  auto f = [](const auto& x) { return x[0]; };
  EXPECT_THROW(a::hill_climb_run(f, {2}, {1}), std::invalid_argument);
  a::HillClimbOptions o;
  o.max_evaluations = 1;
  EXPECT_THROW(a::hill_climb_run(f, {0}, {1}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(a::hill_climb_trace(f, {0}, {1}, o), std::length_error);
  EXPECT_EQ(a::hill_objective_grid(f, 0, 1, .5).values.size(), 3);
}
TEST(Batch314Range, RandomRoundTripsAndRegisterHistories) {
  std::mt19937 gen(17);
  for (int trial = 0; trial < 150; ++trial) {
    std::vector<std::uint32_t> f{1, 2, 3, 7};
    std::vector<int>           x(trial);
    for (auto& v : x)
      v = gen() % 4;
    auto e = a::range_encode_trace(x, f);
    auto d = a::range_decode_trace(e.bytes, f, x.size());
    EXPECT_EQ(d.symbols, x);
    EXPECT_EQ(e.steps.size(), x.size());
    EXPECT_EQ(d.steps.size(), x.size());
    EXPECT_EQ(e.bytes, a::range_encode(x, f));
    std::size_t shifts = 0;
    for (const auto& s : e.steps) {
      shifts += s.shifts.size();
      EXPECT_GT(s.unit, 0);
      EXPECT_GT(s.range_after, 0);
    }
    EXPECT_EQ(shifts, e.renormalizations);
    EXPECT_EQ(e.bytes.size(), shifts + 4);
  }
}
TEST(Batch314Range, InvalidModelTruncationAndLimits) {
  EXPECT_THROW(a::range_encode({1}, {1, 0}), std::invalid_argument);
  EXPECT_THROW(a::range_encode({0}, {65537}), std::invalid_argument);
  EXPECT_THROW(a::range_decode({0, 0, 0}, {1}, 1), std::runtime_error);
  EXPECT_EQ(a::range_encode({}, {1}).size(), 4);
  a::RangeCodingOptions o;
  o.max_trace_entries = 1;
  EXPECT_THROW(a::range_encode_trace({0, 0}, {1}, o), std::length_error);
}
TEST(Batch314Ransac, RefitMaskRepresentsReturnedLine) {
  auto p = a::ransac_example_points();
  auto r = a::ransac_line_trace(p.x, p.y, .15, 200, 3);
  ASSERT_TRUE(r.found);
  EXPECT_TRUE(r.refitted);
  std::size_t count = 0;
  for (std::size_t i = 0; i < p.x.size(); ++i) {
    bool inside = std::abs(r.line.a * p.x[i] + r.line.b * p.y[i] + r.line.c) <= .15;
    EXPECT_EQ(bool(r.line.inliers[i]), inside);
    count += inside;
  }
  EXPECT_EQ(count, r.line.inlier_count);
  EXPECT_EQ(r.trials.size(), 200);
  EXPECT_EQ(r.draws, 400);
  EXPECT_EQ(a::ransac_required_iterations(.67, 2), 8);
  EXPECT_EQ(a::ransac_pair_iterations(2, 2), 7);
}
TEST(Batch314Ransac, DegenerateAndInvalidInputs) {
  EXPECT_FALSE(a::ransac_line_run({1, 1}, {2, 2}, .1, 20).found);
  EXPECT_FALSE(a::ransac_line_run({0, 1}, {0, 1}, .1, 0).found);
  EXPECT_THROW(a::ransac_line_run({0}, {}, .1, 1), std::invalid_argument);
  EXPECT_THROW(a::ransac_line_run({0, 1}, {0, 1}, -.1, 1), std::invalid_argument);
  EXPECT_THROW(a::ransac_required_iterations(0, 2), std::domain_error);
}
TEST(Batch314Rms, SparseIdsAndBoundaryWork) {
  std::vector<a::RTTask> tasks{{7, 4, 1}, {91, 6, 2}};
  auto                   t = a::rt_trace(tasks, 24, a::RTPolicy::rate_monotonic);
  EXPECT_EQ(t.tick_jobs.size(), 24);
  EXPECT_EQ(t.work.ticks, 24);
  EXPECT_EQ(t.work.busy_ticks + t.work.idle_ticks, 24);
  EXPECT_EQ(a::rate_monotonic_priorities(tasks), (std::vector<std::size_t>{7, 91}));
}
TEST(Batch314Rayleigh, MatchingResidualAndExtremeStart) {
  auto r = a::rayleigh_iteration_trace({{1, 0}, {0, 3}}, {1e300, 2e299});
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.pair.value, 1, 1e-12);
  EXPECT_NEAR(std::hypot(r.pair.vector[0], r.pair.vector[1]), 1, 1e-14);
  EXPECT_EQ(r.steps.size(), r.solves + 1);
  EXPECT_EQ(r.matrix_vector_products, 1 + 2 * r.solves);
  EXPECT_DOUBLE_EQ(r.residual, r.steps.back().residual);
  EXPECT_GT(r.solves, 0);
}
TEST(Batch314Rayleigh, ExactEigenvectorZeroBudgetAndSingularShift) {
  auto exact = a::rayleigh_iteration_trace({{1, 0}, {0, 3}}, {1, 0});
  EXPECT_TRUE(exact.converged);
  EXPECT_EQ(exact.solves, 0);
  EXPECT_EQ(exact.steps.size(), 1);
  a::RayleighOptions o;
  o.max_iterations = 0;
  auto r           = a::rayleigh_iteration_run({{1, 0}, {0, 3}}, {1, 1}, o);
  EXPECT_FALSE(r.converged);
  EXPECT_NEAR(r.pair.value, 2, 1e-14);
  EXPECT_THROW(
      a::rayleigh_iteration_run({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, {1, 0, 1}), std::domain_error);
  EXPECT_THROW(a::rayleigh_iteration_run({{1, 1}, {0, 2}}, {1, 0}), std::invalid_argument);
}
TEST(Batch314Rbf, InterceptAndIndependentFixedBasis) {
  auto intercept = a::rbf_train_run({{0}, {1}}, {2, 4}, 0, 0);
  EXPECT_NEAR(a::rbf_predict(intercept.model, {0}), 3, 1e-14);
  auto model = a::rbf_fit_output({{0}, {1}, {2}}, {1, 2, 1}, {{0}, {2}}, 1, 1e-4);
  EXPECT_NEAR(a::rbf_predict(model, {0}), a::rbf_predict(model, {2}), 1e-12);
  auto t = a::rbf_train_trace({{0}, {1}, {2}}, {1, 2, 1}, 2, 1e-3, 1, 3, 1);
  EXPECT_EQ(t.passes.size(), 3);
  EXPECT_EQ(t.design.size(), 3);
  EXPECT_THROW(a::rbf_train({{0}, {1}}, {2}, 1), std::invalid_argument);
}
TEST(Batch314TensorRbf, ValuesAndAnalyticGradients) {
  auto x = n::from_vector(std::vector<double>{0, 1}, {2, 1});
  auto c = n::from_vector(std::vector<double>{0, 2}, {2, 1});
  x.requires_grad_();
  c.requires_grad_();
  auto phi    = n::rbf_features(x, c, 1);
  auto grads  = n::grad(n::sum(phi), {x, c});
  auto values = phi.to_vector<double>();
  EXPECT_NEAR(values[0], 1, 1e-14);
  EXPECT_NEAR(values[1], std::exp(-2), 1e-14);
  auto gx = grads[0].to_vector<double>();
  auto gc = grads[1].to_vector<double>();
  EXPECT_NEAR(gx[0], 2 * std::exp(-2), 1e-12);
  EXPECT_NEAR(gx[1], 0, 1e-12);
  EXPECT_NEAR(
      std::accumulate(gx.begin(), gx.end(), 0.) + std::accumulate(gc.begin(), gc.end(), 0.),
      0,
      1e-12);
  EXPECT_THROW(n::rbf_features(x, c, 0), std::invalid_argument);
  EXPECT_THROW(n::rbf_features(x, c, 1, 3), std::length_error);
}
TEST(Batch314Rc4, OfficialRfcVectorAndActualSwapProducer) {
  std::string               key{"\x01\x02\x03\x04\x05", 5};
  std::vector<std::uint8_t> expected{
      0xb2,
      0x39,
      0x63,
      0x05,
      0xf0,
      0x3d,
      0xc0,
      0x27,
      0xcc,
      0xc3,
      0x52,
      0x4a,
      0x0a,
      0x11,
      0x18,
      0xa8};
  auto t = a::rc4_trace(key, 16);
  EXPECT_EQ(t.bytes, expected);
  EXPECT_EQ(t.scheduling.size(), 256);
  EXPECT_EQ(t.steps.size(), 16);
  a::Rc4 stream(key);
  for (auto byte : expected)
    EXPECT_EQ(stream.next_step().output, byte);
  EXPECT_EQ(stream.state().permutation, t.final.permutation);
  EXPECT_THROW(a::rc4_keystream("", 0), std::invalid_argument);
  EXPECT_EQ(
      a::rc4_encrypt("Key", a::rc4_encrypt("Key", std::string("a\0b", 3))), std::string("a\0b", 3));
}
TEST(Batch314Parser, ActualTreeAndDecimalGrammar) {
  auto t = a::recursive_descent_trace("1+2*3");
  EXPECT_EQ(t.value, 7);
  ASSERT_LT(t.root, t.nodes.size());
  EXPECT_EQ(t.nodes[t.root].value, 7);
  EXPECT_EQ(t.nodes.size(), 5);
  EXPECT_NO_THROW(a::recursive_descent_diagram(t));
  EXPECT_EQ(a::recursive_descent_eval("--.5+(4-1)/2"), 2);
  for (const std::string s :
       std::vector<std::string>{"1.2.3", "1e2", "(1", "1/0", std::string("1\0+2", 4)})
    EXPECT_THROW(a::recursive_descent_eval(s), std::runtime_error);
  a::RecursiveDescentOptions o;
  o.max_nodes = 1;
  EXPECT_THROW(a::recursive_descent_trace("1+2", o), std::length_error);
}
TEST(Batch314Refcount, CycleHeldTailAndRealLifoFree) {
  auto cycle = a::reference_counting_trace({3, {{1}, {0, 2}, {}}, {}, {}});
  EXPECT_EQ(cycle.result.leaked, (std::vector<int>{0, 1, 2}));
  EXPECT_TRUE(cycle.freed_order.empty());
  auto chain = a::reference_counting_trace(a::reference_count_chain(5));
  EXPECT_EQ(chain.freed_order, (std::vector<int>{0, 1, 2, 3, 4}));
  EXPECT_EQ(chain.increments, 4);
  EXPECT_EQ(chain.decrements, 4);
  EXPECT_EQ(chain.final_counts, (std::vector<int>{0, 0, 0, 0, 0}));
  EXPECT_THROW(a::reference_counting_run({1, {{-1}}, {}, {}}), std::invalid_argument);
}
TEST(Batch314Division, FullWidthIdentityAndActualWork) {
  std::mt19937_64 gen(77);
  for (int i = 0; i < 200; ++i) {
    auto N = gen(), D = gen() | 1;
    auto t = a::restoring_division_trace(N, D);
    EXPECT_EQ(t.analysis.result.quotient, N / D);
    EXPECT_EQ(t.analysis.result.remainder, N % D);
    EXPECT_EQ(t.steps.size(), 64);
    EXPECT_EQ(t.analysis.comparison_tests, 64);
    EXPECT_EQ(t.analysis.additions, 0);
  }
  auto t = a::restoring_division_trace(217, 13);
  EXPECT_EQ(t.analysis.result.quotient, 16);
  EXPECT_EQ(t.analysis.result.remainder, 9);
  EXPECT_EQ(t.analysis.subtractions, 1);
  EXPECT_FALSE(a::restoring_division_trace(1, 0).analysis.valid_divisor);
}
