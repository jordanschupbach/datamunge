#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/ntt_multiplication.hpp>
#include <datamunge/algorithms/rvm.hpp>
#include <datamunge/algorithms/salsa20.hpp>
#include <datamunge/algorithms/samplesort.hpp>
#include <datamunge/algorithms/sarsa.hpp>
#include <datamunge/algorithms/scanline_fill.hpp>
#include <datamunge/algorithms/schensted.hpp>
#include <datamunge/algorithms/schreier_sims.hpp>
#include <datamunge/algorithms/secant.hpp>
#include <datamunge/algorithms/shortest_common_supersequence.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
TEST(Batch334RVM, InitialPosteriorAndMatchedCovariance) {
  a::RVMOptions o;
  o.max_iterations = 0;
  auto r           = a::rvm_train_trace({{0}}, {1}, o);
  ASSERT_EQ(r.posterior_solves, 1);
  const double b    = r.model.beta;
  const double mean = 2 * b / (1 + 2 * b);
  EXPECT_NEAR(a::rvm_predict(r.model, {0}), mean, 1e-9);
  auto p = a::rvm_predictive(r, {0});
  EXPECT_NEAR(p.latent_variance, 2 / (1 + 2 * b), 1e-12);
  EXPECT_DOUBLE_EQ(p.noise_variance, 1 / b);
  EXPECT_DOUBLE_EQ(p.variance, p.latent_variance + p.noise_variance);
  EXPECT_TRUE(r.steps.empty());
}
TEST(Batch334RVM, FinalActiveNormalEquations) {
  auto          e = a::rvm_sine_example();
  a::RVMOptions o;
  o.gamma          = .5;
  o.max_iterations = 30;
  auto r           = a::rvm_train_trace(e.inputs, e.targets, o);
  ASSERT_EQ(r.posterior_solves, r.steps.size() + 1);
  ASSERT_EQ(r.alpha.size(), 42);
  for (std::size_t j = 0; j < 42; ++j) {
    if (j && !r.model.relevant[j - 1]) {
      EXPECT_EQ(r.model.weights[j], 0);
      continue;
    }
    long double residual = r.alpha[j] * r.model.weights[j];
    for (std::size_t i = 0; i < e.inputs.size(); ++i) {
      const double phi = j ? std::exp(-.5 * std::pow(e.inputs[i][0] - e.inputs[j - 1][0], 2)) : 1;
      residual += r.model.beta * phi * (a::rvm_predict(r.model, e.inputs[i]) - e.targets[i]);
    }
    EXPECT_NEAR(double(residual), 0, 1e-5);
  }
  for (const auto& x : e.grid) {
    auto p = a::rvm_predictive(r, x);
    EXPECT_GE(p.latent_variance, 0);
    EXPECT_TRUE(std::isfinite(p.variance));
  }
}
TEST(Batch334RVM, ShapeFiniteAndPolicies) {
  a::RVMOptions initial; initial.max_iterations=0;
  EXPECT_THROW(a::rvm_train_run({{0},{1}},{-1e308,1e308},initial),std::domain_error);
  EXPECT_THROW(a::rvm_train({}, {}), std::invalid_argument);
  EXPECT_THROW(a::rvm_train({{0}, {1, 2}}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(a::rvm_train({{0}}, {0}, -1), std::invalid_argument);
  a::RVMOptions o;
  o.max_trace_entries = 1;
  EXPECT_THROW(a::rvm_train_trace({{0}}, {0}, o), std::length_error);
  EXPECT_THROW(a::rvm_predict(a::RVMModel{}, {}), std::invalid_argument);
}
TEST(Batch334Salsa, OfficialAndQuarterChain) {
  auto e = a::salsa20_example();
  auto r = a::salsa20_core_trace(e.input);
  EXPECT_EQ(r.bytes, e.expected);
  ASSERT_EQ(r.quarters.size(), 80);
  auto state = r.initial;
  for (const auto& q : r.quarters) {
    for (int i = 0; i < 4; ++i)
      EXPECT_EQ(q.before[i], state[q.indices[i]]);
    for (int i = 0; i < 4; ++i)
      state[q.indices[i]] = q.after[i];
  }
  EXPECT_EQ(state, r.mixed);
  for (int i = 0; i < 16; ++i)
    EXPECT_EQ(r.output_words[i], std::uint32_t(r.initial[i] + r.mixed[i]));
}
TEST(Batch334Salsa, RawBoundaryLengthsAndCounterAdmission) {
  auto e = a::salsa20_example();
  for (std::size_t n : {0, 1, 63, 64, 65, 145}) {
    std::string x(n, char{0});
    for (std::size_t i = 0; i < n; ++i)
      x[i] = char(i);
    auto r = a::salsa20_stream_trace(e.key, 7, e.nonce, x);
    EXPECT_EQ(r.block_calls, n / 64 + (n % 64 != 0));
    EXPECT_EQ(r.byte_xors, n);
    EXPECT_EQ(a::salsa20_encrypt(e.key, 7, e.nonce, r.output), x);
  }
  EXPECT_NO_THROW(a::salsa20_encrypt({}, UINT64_MAX, {}, "a"));
  EXPECT_THROW(a::salsa20_encrypt({}, UINT64_MAX, {}, std::string(65, 'a')), std::overflow_error);
  EXPECT_THROW(a::salsa20_core_trace(e.input, 767), std::length_error);
}
TEST(Batch334Sample, SortedPermutationAndCounts) {
  std::mt19937 rng(19);
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<int> x(trial * 7);
    for (auto& v : x)
      v = int(rng() % 31) - 15;
    auto expected = x;
    std::sort(expected.begin(), expected.end());
    auto r = a::samplesort_trace(x);
    EXPECT_EQ(r.values, expected);
    EXPECT_EQ(r.comparisons, r.sample_comparisons + r.splitter_comparisons + r.leaf_comparisons);
    EXPECT_EQ(r.sample_reads, 24 * r.partitions);
    for (const auto& s : r.steps)
      if (s.stage == a::SampleSortStage::Partition) {
        EXPECT_EQ(
            std::accumulate(s.bucket_sizes.begin(), s.bucket_sizes.end(), std::size_t{}),
            s.before.size());
        EXPECT_EQ(s.sample.size(), 24);
        EXPECT_EQ(s.splitters.size(), 7);
      }
  }
}
TEST(Batch334Sample, FallbackDepthAndInvalidKeys) {
  auto r = a::samplesort_trace(std::vector<int>(100, 7));
  EXPECT_EQ(r.partitions, 1);
  EXPECT_EQ(r.fallbacks, 1);
  a::SampleSortOptions o;
  o.depth_limit = 1;
  auto t        = a::samplesort_run(a::samplesort_example_values(1000), o);
  EXPECT_TRUE(std::is_sorted(t.values.begin(), t.values.end()));
  EXPECT_LE(t.maximum_depth, 1);
  o.max_comparisons = 1;
  EXPECT_THROW(a::samplesort_run(std::vector<int>{3, 2, 1}, o), std::length_error);
  EXPECT_THROW(a::samplesort_run(std::vector<double>{NAN}), std::invalid_argument);
}
TEST(Batch334Sarsa, ActualSampledUpdateAndSelections) {
  a::TabularEnvironment e{
      2,
      1,
      [] { return std::size_t{0}; },
      [](std::size_t s, std::size_t) { return a::StepResult{1, 2, s == 1}; }};
  a::RLParameters p;
  p.episodes      = 1;
  p.max_steps     = 2;
  p.learning_rate = .5;
  p.discount      = 1;
  p.epsilon       = 0;
  auto r          = a::sarsa_trace(e, p);
  ASSERT_EQ(r.steps.size(), 2);
  EXPECT_DOUBLE_EQ(r.steps[0].after, 1);
  EXPECT_DOUBLE_EQ(r.steps[1].after, 1);
  EXPECT_EQ(r.action_selections, 2);
  EXPECT_TRUE(r.steps[1].terminal);
  EXPECT_EQ(r.steps[1].bootstrap, 0);
  p.max_steps = 0;
  auto zero   = a::sarsa_trace(e, p);
  EXPECT_EQ(zero.transitions, 0);
  EXPECT_EQ(zero.action_selections, 1);
  p.max_steps = 1;
  auto cap    = a::sarsa_trace(e, p);
  EXPECT_EQ(cap.action_selections, 2);
  EXPECT_FALSE(cap.terminated[0]);
}
TEST(Batch334Sarsa, InvalidCallbacksAndCliff) {
  auto e     = a::tabular_cliff();
  auto cliff = e.step(36, 1);
  EXPECT_EQ(cliff.next_state, 36);
  EXPECT_EQ(cliff.reward, -100);
  EXPECT_FALSE(cliff.terminal);
  EXPECT_TRUE(e.step(35, 2).terminal);
  EXPECT_THROW(e.step(48, 0), std::invalid_argument);
  e.step = [](auto, auto) { return a::StepResult{100, 0, true}; };
  EXPECT_THROW(a::sarsa_run(e, a::RLParameters{}), std::invalid_argument);
  a::RLParameters p;
  p.learning_rate = 0;
  EXPECT_THROW(a::sarsa_run(a::tabular_cliff(), p), std::invalid_argument);
}
TEST(Batch334Scan, OriginalAndFractionalMinima) {
  EXPECT_EQ(a::scanline_fill({{0, 0}, {10, 0}, {10, 10}, {0, 10}}).size(), 100);
  EXPECT_EQ(a::scanline_fill({{0, 0}, {20, 0}, {0, 20}}).size(), 210);
  auto t = a::scanline_fill_trace({{.2, .2}, {2.2, .2}, {2.2, 2.2}, {.2, 2.2}});
  ASSERT_EQ(t.pixels.size(), 4);
  EXPECT_EQ(t.rows.front().y, 0);
  std::set<std::pair<int, int>> unique;
  for (auto p : t.pixels)
    unique.insert({p.x, p.y});
  EXPECT_EQ(unique.size(), t.pixels.size());
}
TEST(Batch334Scan, ExtremeClipAndPolicies) {
  a::ScanlineOptions o;
  o.min_x         = INT_MAX;
  o.max_x         = INT_MAX;
  o.min_y         = INT_MAX;
  o.max_y         = INT_MAX;
  const double a1 = double(INT_MAX) - 2, b = double(INT_MAX) + 2;
  auto         r = a::scanline_fill_run({{a1, a1}, {b, a1}, {b, b}, {a1, b}}, o);
  ASSERT_EQ(r.pixels.size(), 1);
  EXPECT_EQ(r.pixels[0], (a::Pixel{INT_MAX, INT_MAX}));
  o            = {};
  o.max_pixels = 3;
  EXPECT_THROW(a::scanline_fill_run({{0, 0}, {2, 0}, {2, 2}, {0, 2}}, o), std::length_error);
  EXPECT_THROW(a::scanline_fill({{0, NAN}}), std::invalid_argument);
}
TEST(Batch334Schensted, OriginalAndDuplicateConvention) {
  auto r = a::schensted_trace({4, 1, 3, 6, 2, 5});
  EXPECT_EQ(r.tableau.p, (std::vector<std::vector<int>>{{1, 2, 5}, {3, 6}, {4}}));
  EXPECT_EQ(r.appends, 6);
  std::size_t bumps = 0;
  for (const auto& s : r.steps) {
    if (s.bumped) {
      ++bumps;
      EXPECT_GT(s.outgoing, s.carried);
    } else
      EXPECT_EQ(s.after.size(), s.before.size() + 1);
  }
  EXPECT_EQ(bumps, r.bumps);
  auto duplicate = a::schensted_run({1, 1, 2, 2});
  EXPECT_EQ(duplicate.tableau.p.size(), 1);
  EXPECT_EQ(duplicate.tableau.p[0].size(), 4);
}
TEST(Batch334Schensted, EmptyAndBudgets) {
  EXPECT_TRUE(a::rsk_insert({}).p.empty());
  a::SchenstedOptions o;
  o.max_comparisons = 1;
  EXPECT_THROW(a::schensted_run({1, 2, 3}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(a::schensted_trace({1}, o), std::length_error);
}
TEST(Batch334NTT, ExactSmallProductsAndActualCarry) {
  for (std::uint64_t i = 0; i < 100; ++i) {
    auto r = a::ntt_multiply_trace(std::to_string(i * 79), std::to_string(i + 17));
    EXPECT_EQ(r.product, std::to_string(i * 79 * (i + 17)));
    for (const auto& s : r.carries) {
      EXPECT_EQ(s.digit, (s.coefficient + s.before) % 10);
      EXPECT_EQ(s.after, (s.coefficient + s.before) / 10);
    }
    if (r.transform_size) {
      EXPECT_EQ(r.pointwise_products, r.transform_size);
      EXPECT_EQ(r.carry_updates, r.transform_size);
      std::size_t log = 0;
      for (auto n = r.transform_size; n > 1; n /= 2)
        ++log;
      EXPECT_EQ(r.butterflies, 3 * r.transform_size * log / 2);
    }
  }
}
TEST(Batch334NTT, ValidationLeadingZerosAndPolicies) {
  EXPECT_EQ(a::ntt_mul("00012", "034"), "408");
  EXPECT_EQ(a::ntt_mul("0", "999"), "0");
  EXPECT_THROW(a::ntt_mul("0", "bad"), std::invalid_argument);
  EXPECT_THROW(a::ntt_mul("", "2"), std::invalid_argument);
  a::NTTMultiplicationOptions o;
  o.max_transform_size = 2;
  EXPECT_THROW(a::ntt_multiply_run("12", "34", o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(a::ntt_multiply_trace("1", "2", o), std::length_error);
}
TEST(Batch334Schreier, SymmetricOrdersAndEverySmallCandidate) {
  unsigned long long factorial = 1;
  for (int n = 2; n <= 9; ++n) {
    factorial *= n;
    if (n == 2)
      factorial = 2;
    auto r = a::schreier_sims_trace(n, a::symmetric_group_generators(n));
    EXPECT_EQ(r.group.order, factorial);
    if (n <= 5) {
      std::vector<int> p(n);
      std::iota(p.begin(), p.end(), 0);
      do {
        EXPECT_TRUE(a::bsgs_contains(r.group, p));
      } while (std::next_permutation(p.begin(), p.end()));
    }
  }
  auto cyclic = a::schreier_sims(4, {{1, 2, 3, 0}});
  EXPECT_EQ(cyclic.order, 4);
  EXPECT_TRUE(a::bsgs_contains(cyclic, {2, 3, 0, 1}));
  EXPECT_FALSE(a::bsgs_contains(cyclic, {1, 0, 2, 3}));
}
TEST(Batch334Schreier, MalformedInputsAndWorkAdmission) {
  EXPECT_EQ(a::schreier_sims(0, {}).order, 1);
  EXPECT_THROW(a::schreier_sims(-1, {}), std::invalid_argument);
  EXPECT_THROW(a::schreier_sims(3, {{0, 0, 2}}), std::invalid_argument);
  a::SchreierSimsOptions o;
  o.max_point_operations = 1;
  EXPECT_THROW(a::schreier_sims_run(3, {{1, 2, 0}}, o), std::length_error);
  auto g    = a::schreier_sims(3, {{1, 2, 0}});
  g.base[0] = 4;
  EXPECT_THROW(a::bsgs_contains(g, {0, 1, 2}), std::invalid_argument);
}
TEST(Batch334SCS, ActualDpAndShortestContainment) {
  auto r = a::shortest_common_supersequence_trace({1, 2}, {2, 3});
  EXPECT_EQ(r.sequence, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(r.lcs_length, 1);
  EXPECT_EQ(r.sequence.size(), 4 - r.lcs_length);
  EXPECT_EQ(r.table.back().back(), 1);
  auto rev = r.sequence;
  std::reverse(rev.begin(), rev.end());
  ASSERT_EQ(rev.size(), r.steps.size());
  for (std::size_t i = 0; i < rev.size(); ++i)
    EXPECT_EQ(rev[i], r.steps[i].value);
  EXPECT_EQ(a::shortest_common_supersequence({}, {1, 2}), (std::vector<int>{1, 2}));
}
TEST(Batch334SCS, TieOrderingAndTableCap) {
  EXPECT_EQ(a::shortest_common_supersequence({1}, {2}), (std::vector<int>{2, 1}));
  a::SCSOptions o;
  o.max_cells = 3;
  EXPECT_THROW(a::shortest_common_supersequence_run({1}, {2}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(a::shortest_common_supersequence_trace({1}, {1}, o), std::length_error);
}
TEST(Batch334Secant, ActualQueriesAndExplicitStops) {
  std::vector<double> queries;
  auto                r = a::secant_trace(
      [&](double x) {
        queries.push_back(x);
        return x * x - 2;
      },
      1,
      2);
  EXPECT_TRUE(r.result.converged);
  EXPECT_NEAR(r.result.root, std::sqrt(2.), 1e-9);
  ASSERT_EQ(queries.size(), r.evaluations.size());
  for (std::size_t i = 0; i < queries.size(); ++i)
    EXPECT_EQ(queries[i], r.evaluations[i].x);
  EXPECT_EQ(r.function_evaluations, std::size_t(r.result.iterations) + 2);
  auto flat = a::secant_trace([](double) { return 1.; }, 0, 1);
  EXPECT_EQ(flat.stop, a::SecantStop::Horizontal);
  EXPECT_FALSE(flat.result.converged);
  EXPECT_EQ(flat.function_evaluations, 2);
}
TEST(Batch334Secant, InitialSuccessBudgetAndInvalidCallbacks) {
  auto exact = a::secant_run([](double x) { return x; }, 0, 1);
  EXPECT_TRUE(exact.result.converged);
  EXPECT_EQ(exact.result.root, 0);
  EXPECT_EQ(exact.function_evaluations, 2);
  a::SecantOptions o;
  o.max_iterations = 0;
  auto r           = a::secant_run([](double x) { return x * x - 2; }, 1, 2, o);
  EXPECT_EQ(r.stop, a::SecantStop::Budget);
  EXPECT_FALSE(r.result.converged);
  EXPECT_THROW(a::secant_run([](double) { return NAN; }, 0, 1), std::invalid_argument);
  o                 = {};
  o.max_evaluations = 1;
  EXPECT_THROW(a::secant_run([](double x) { return x; }, 0, 1, o), std::length_error);
}
