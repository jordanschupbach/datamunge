#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/srt_division.hpp>
#include <datamunge/algorithms/sss_star.hpp>
#include <datamunge/algorithms/steinhaus_johnson_trotter.hpp>
#include <datamunge/algorithms/stochastic_universal_sampling.hpp>
#include <datamunge/algorithms/stone_sip.hpp>
#include <datamunge/algorithms/stooge_sort.hpp>
#include <datamunge/algorithms/strand_sort.hpp>
#include <datamunge/algorithms/strassen.hpp>
#include <datamunge/algorithms/structured_svm_analysis.hpp>
#include <datamunge/algorithms/subgraph_isomorphism.hpp>
#include <limits>
#include <numeric>
#include <set>
namespace a = datamunge::algorithms;
TEST(Batch374, SrtActualRecurrence) {
  auto   r    = a::srt_division_trace(.7, .9, 40);
  double prev = r.normalized_numerator, q = 0;
  for (const auto& s : r.steps) {
    EXPECT_EQ(s.before, prev);
    EXPECT_EQ(s.shifted, 2 * prev);
    EXPECT_EQ(s.digit, s.shifted >= .5 ? 1 : s.shifted <= -.5 ? -1 : 0);
    EXPECT_EQ(s.after, s.shifted - s.digit * r.normalized_divisor);
    q += s.digit * s.weight;
    EXPECT_EQ(q, s.value);
    prev = s.after;
  }
  EXPECT_EQ(q, r.value);
  EXPECT_NEAR(q, .7 / .9, std::ldexp(1., -40));
}
TEST(Batch374, SrtValidation) {
  EXPECT_THROW(a::srt_divide(1, 0), std::invalid_argument);
  EXPECT_THROW(a::srt_divide(1, 1), std::invalid_argument);
  EXPECT_THROW(a::srt_divide(.1, 1, -1), std::invalid_argument);
  EXPECT_THROW(a::srt_digits_to_value({2}), std::invalid_argument);
  EXPECT_EQ(a::srt_divide(0, 1, 0), 0);
  EXPECT_DOUBLE_EQ(a::srt_divide(.5, 1, 4), .5);
}
TEST(Batch374, SssMatchesFullSearch) {
  for (int d = 0; d <= 5; ++d)
    for (unsigned seed = 0; seed < 10; ++seed) {
      auto t = a::sss_example_tree(d, seed);
      for (bool max : {false, true}) {
        auto r   = a::sss_star_trace(t, max);
        auto ref = a::minimax(t, max);
        ASSERT_TRUE(r.converged);
        EXPECT_EQ(r.result.value, ref.value);
        EXPECT_EQ(r.preflight_leaf_reads, std::size_t(ref.leaves_evaluated));
        EXPECT_EQ(r.probes, r.history.size());
        EXPECT_EQ(r.lower, r.upper);
      }
    }
}
TEST(Batch374, SssLimitsAndExtreme) {
  auto          t = a::sss_example_tree(2, 1);
  a::SSSOptions o;
  o.max_probes = 0;
  auto r       = a::sss_star_run(t, true, 1, o);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.result.leaves_evaluated, 0);
  a::GameNode e{
      false,
      0,
      {{true, -std::numeric_limits<double>::max(), {}},
       {true, std::numeric_limits<double>::max(), {}}}};
  EXPECT_EQ(a::sss_star_run(e, true).result.value, std::numeric_limits<double>::max());
  a::GameNode bad{true, 0, {{true, 1, {}}}};
  EXPECT_THROW(a::sss_star_run(bad, true), std::invalid_argument);
  a::GameNode gap{false, 0, {{true, .1, {}}, {true, .2, {}}}};
  EXPECT_THROW(a::sss_star_run(gap, true, 1), std::invalid_argument);
}
TEST(Batch374, SjtCompleteAdjacentStream) {
  for (std::size_t n = 0; n <= 7; ++n) {
    auto                       r = a::sjt_trace(n);
    std::set<std::vector<int>> unique(r.permutations.begin(), r.permutations.end());
    std::size_t                factorial = 1;
    for (std::size_t j = 1; j <= n; ++j)
      factorial *= j;
    EXPECT_EQ(unique.size(), factorial);
    EXPECT_EQ(r.mobile_tests, n * factorial);
    EXPECT_EQ(r.adjacent_swaps, factorial - 1);
    for (std::size_t j = 1; j < factorial; ++j) {
      const auto& s   = r.history[j];
      auto        old = r.permutations[j - 1];
      ASSERT_EQ(std::abs(s.to - s.from), 1);
      EXPECT_EQ(old[s.from], s.mobile);
      std::swap(old[s.from], old[s.to]);
      EXPECT_EQ(old, r.permutations[j]);
    }
    for (std::size_t j = 0; j < factorial; ++j)
      for (std::size_t v = 1; v <= n; ++v)
        EXPECT_EQ(r.permutations[j][r.value_positions[v][j]], int(v));
  }
}
TEST(Batch374, SjtPreflightLimits) {
  a::SJTOptions o;
  o.max_permutations = 5;
  EXPECT_THROW(a::sjt_run(3, o), std::length_error);
  EXPECT_THROW(a::sjt_run(100), std::length_error);
  o                    = {};
  o.max_history_values = 2;
  EXPECT_THROW(a::sjt_trace(3, o), std::length_error);
  EXPECT_EQ(a::sjt_run(0).permutations, std::vector<std::vector<int>>{{}});
}
TEST(Batch374, SusActualPointers) {
  for (unsigned seed = 0; seed < 100; ++seed) {
    auto r = a::sus_trace({0, 1, 2, 3, 0}, 17, seed);
    EXPECT_EQ(r.random_words, 1);
    EXPECT_EQ(r.counts[0] + r.counts[4], 0);
    EXPECT_EQ(std::accumulate(r.counts.begin(), r.counts.end(), std::size_t{}), 17);
    for (const auto& s : r.pointers) {
      EXPECT_EQ(s.selected, r.indices[s.pick]);
      EXPECT_LT(s.position, r.cumulative[s.selected]);
      if (s.selected)
        EXPECT_GE(s.position, r.cumulative[s.selected - 1]);
    }
    for (std::size_t i = 0; i < 5; ++i) {
      EXPECT_GE(r.counts[i], std::floor(r.expected[i]));
      EXPECT_LE(r.counts[i], std::ceil(r.expected[i]));
    }
  }
}
TEST(Batch374, SusChecksAndNoDraw) {
  EXPECT_EQ(a::sus_run({0, 0}, 0).random_words, 0);
  EXPECT_THROW(a::sus_run({0, 0}, 1), std::invalid_argument);
  EXPECT_THROW(a::sus_run({1, -1}, 0), std::invalid_argument);
  EXPECT_THROW(a::sus_run({1}, -1), std::invalid_argument);
  EXPECT_THROW(a::sus_run({1, std::numeric_limits<double>::infinity()}, 0), std::invalid_argument);
  EXPECT_EQ(a::sus_run({1, 1}, 4, 3).counts, (std::vector<std::size_t>{2, 2}));
}
TEST(Batch374, StoneMatchingResidual) {
  auto            g = a::stone_poisson_example(5, 4);
  a::StoneOptions o;
  o.max_iterations = 1;
  auto r           = a::stone_sip_trace(g, o);
  ASSERT_EQ(r.result.iterations, 1);
  ASSERT_EQ(r.history.size(), 2);
  double norm = 0;
  for (int y = 0; y < g.ny; ++y)
    for (int x = 0; x < g.nx; ++x) {
      int    p = y * g.nx + x;
      double z = g.Ap[p] * r.result.x[p];
      if (x)
        z += g.Aw[p] * r.result.x[p - 1];
      if (x + 1 < g.nx)
        z += g.Ae[p] * r.result.x[p + 1];
      if (y)
        z += g.As[p] * r.result.x[p - g.nx];
      if (y + 1 < g.ny)
        z += g.An[p] * r.result.x[p + g.nx];
      norm = std::hypot(norm, g.b[p] - z);
    }
  EXPECT_DOUBLE_EQ(norm, r.result.residual);
  EXPECT_EQ(r.history.back().x, r.result.x);
  EXPECT_EQ(r.history.back().residual, r.result.residual);
  EXPECT_EQ(r.triangular_cells, 40);
}
TEST(Batch374, StoneCapsAndValidation) {
  auto            g = a::stone_poisson_example(4, 4);
  a::StoneOptions o;
  o.max_iterations = 0;
  auto z           = a::stone_sip_run(g, o);
  EXPECT_EQ(z.result.iterations, 0);
  EXPECT_GT(z.result.residual, 0);
  EXPECT_EQ(z.factorized_cells, 16);
  g.Aw[0] = 1;
  EXPECT_THROW(a::stone_sip_run(g), std::invalid_argument);
  g       = a::stone_poisson_example(1, 1);
  g.Ap[0] = 0;
  EXPECT_THROW(a::stone_sip_run(g), std::domain_error);
  EXPECT_TRUE(a::stone_sip_run({}).converged);
}
TEST(Batch374, StoogeExecutesAndSorts) {
  for (int n = 0; n <= 12; ++n) {
    std::vector<int> x(n);
    std::iota(x.rbegin(), x.rend(), 0);
    auto r = a::stooge_sort_trace(x);
    std::sort(x.begin(), x.end());
    EXPECT_EQ(r.values, x);
    EXPECT_EQ(r.steps.size(), r.comparisons);
    if (n > 1)
      EXPECT_EQ(r.steps.back().after, x);
  }
  EXPECT_EQ(a::stooge_sort_run(std::vector<int>{3, 2, 1}).comparisons, 4);
}
TEST(Batch374, StoogeAtomicFailure) {
  std::vector<double> x{1, std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(a::stooge_sort(x), std::invalid_argument);
  EXPECT_EQ(x[0], 1);
  a::StoogeSortOptions o;
  o.max_comparisons = 1;
  EXPECT_THROW(a::stooge_sort_run(std::vector<int>{3, 2, 1}, o), std::length_error);
  EXPECT_EQ(a::stooge_sort_run(std::vector<int>{}).comparisons, 0);
}
struct StableKey {
  int  key, id;
  bool operator<(const StableKey& b) const { return key < b.key; }
};
TEST(Batch374, StrandStableAndHistories) {
  std::vector<StableKey> x{{2, 0}, {1, 1}, {2, 2}, {1, 3}, {2, 4}};
  auto                   r = a::strand_sort_run(x);
  EXPECT_EQ(r.values[0].id, 1);
  EXPECT_EQ(r.values[1].id, 3);
  EXPECT_EQ(r.values[2].id, 0);
  EXPECT_EQ(r.values[3].id, 2);
  EXPECT_EQ(r.values[4].id, 4);
  auto t = a::strand_sort_trace(std::vector<int>{3, 1, 4, 2, 5});
  EXPECT_EQ(t.steps.size(), t.strands);
  EXPECT_EQ(t.steps.back().merged, t.values);
  for (const auto& s : t.steps)
    EXPECT_TRUE(std::is_sorted(s.strand.begin(), s.strand.end()));
}
TEST(Batch374, StrandFixtureAndPolicies) {
  auto f = a::strand_sort_example_arrays();
  ASSERT_EQ(f.size(), 11);
  EXPECT_EQ(a::strand_sort_run(f[0]).strands, 1);
  for (const auto& v : f) {
    auto r = a::strand_sort_run(v);
    EXPECT_TRUE(std::is_sorted(r.values.begin(), r.values.end()));
    EXPECT_EQ(r.values.size(), 400);
  }
  a::StrandSortOptions o;
  o.max_comparisons = 1;
  EXPECT_THROW(a::strand_sort_run(std::vector<int>{3, 2, 1}, o), std::length_error);
  EXPECT_THROW(
      a::strand_sort_run(std::vector<double>{std::numeric_limits<double>::infinity()}),
      std::invalid_argument);
}
TEST(Batch374, StrassenSevenActualProducts) {
  a::StrassenOptions o;
  o.leaf_size = 1;
  auto r      = a::strassen_trace({{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, o);
  EXPECT_EQ(r.product, (a::StrassenMatrix{{19, 22}, {43, 50}}));
  EXPECT_EQ(r.scalar_products, 7);
  EXPECT_EQ(r.leaf_accumulations, 7);
  EXPECT_EQ(r.block_additions, 18);
  EXPECT_EQ(r.recursive_calls, 8);
  EXPECT_EQ(r.steps.size(), 8);
  EXPECT_EQ(r.steps.back().result, r.product);
  EXPECT_EQ(a::strassen_run({{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}).scalar_products, 8);
}
TEST(Batch374, StrassenPaddingRectanglesAndChecks) {
  auto f   = a::strassen_example_matrices();
  auto r   = a::strassen_run(f[0], f[1]);
  auto ref = a::strassen_schoolbook_run(f[0], f[1]);
  EXPECT_EQ(r.padded_order, 128);
  EXPECT_EQ(r.scalar_products, 1835008);
  EXPECT_EQ(ref.scalar_products, 1000000);
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j)
      EXPECT_NEAR(r.product[i][j], ref.product[i][j], 1e-12);
  EXPECT_EQ(a::strassen_run({{1, 2, 3}}, {{1}, {2}, {3}}).product, (a::StrassenMatrix{{14}}));
  EXPECT_THROW(a::strassen_run({{1}, {}}, {{1}}), std::invalid_argument);
  EXPECT_THROW(a::strassen_run({{1, 2}}, {{1}}), std::invalid_argument);
  EXPECT_TRUE(a::strassen_multiply({{1, 2}}, {{1}}).empty());
  a::StrassenOptions o;
  o.max_scalar_products = 1;
  EXPECT_THROW(a::strassen_run({{1, 2}}, {{1}, {2}}, o), std::length_error);
}
TEST(Batch374, SvmSubgradientAndObjective) {
  a::StructuredSVMParameters p;
  p.lambda = 1;
  p.epochs = 1;
  p.seed   = 1;
  auto r   = a::structured_svm_trace({{2}}, {1}, 2, p);
  ASSERT_EQ(r.steps.size(), 1);
  EXPECT_EQ(r.steps[0].truth, 1);
  EXPECT_EQ(r.steps[0].violator, 0);
  EXPECT_EQ(r.model.weights, (std::vector<std::vector<double>>{{-2, -1}, {2, 1}}));
  EXPECT_EQ(r.epochs.back().objective.regularizer, 5);
  EXPECT_EQ(r.epochs.back().objective.mean_hinge, 0);
  EXPECT_EQ(r.epochs.back().objective.correct, 1);
  EXPECT_EQ(a::structured_svm_predict(r.model, {2}), 1);
}
TEST(Batch374, SvmOriginalAndValidation) {
  auto                       d    = a::structured_svm_example_cohort();
  auto                       held = a::structured_svm_example_cohort(60, 11);
  a::StructuredSVMParameters p;
  p.epochs = 100;
  p.seed   = 1;
  auto r   = a::structured_svm_run(d.features, d.labels, 3, p);
  EXPECT_EQ(r.updates, 18000);
  EXPECT_EQ(r.epochs.size(), 101);
  int correct = 0;
  for (std::size_t i = 0; i < held.features.size(); ++i)
    correct += a::structured_svm_predict(r.model, held.features[i]) == std::size_t(held.labels[i]);
  EXPECT_GT(correct, 170);
  auto        grid  = a::structured_svm_regions(r.model, -2, 7, 14);
  std::size_t count = 0;
  for (const auto& x : grid.x)
    count += x.size();
  EXPECT_EQ(count, 196);
  p.epochs = 0;
  EXPECT_EQ(a::structured_svm_run({{1}}, {0}, 1, p).updates, 0);
  EXPECT_THROW(a::structured_svm_predict({}, {}), std::invalid_argument);
  EXPECT_THROW(a::structured_svm_run({{1}}, {2}, 2), std::invalid_argument);
  p.lambda = 0;
  EXPECT_THROW(a::structured_svm_run({{1}}, {0}, 1, p), std::invalid_argument);
}
TEST(Batch374, SubgraphNonInducedAndActualHistory) {
  std::vector<std::vector<int>> p{{1}, {0, 2}, {1}}, t{{1, 2}, {0, 2}, {0, 1}};
  auto                          r = a::subgraph_isomorphism_trace(3, p, 3, t);
  ASSERT_TRUE(r.match.found);
  EXPECT_EQ(r.match.mapping, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(r.steps.back().action, "success");
  EXPECT_EQ(r.steps.back().mapping, r.match.mapping);
  EXPECT_TRUE(a::subgraph_isomorphism_run(0, {}, 0, {}).match.found);
  for (int n = 4; n <= 10; ++n)
    EXPECT_FALSE(a::subgraph_isomorphism_run(3, t, n, a::subgraph_cycle_graph(n)).match.found);
}
TEST(Batch374, SubgraphCompleteValidation) {
  EXPECT_THROW(a::subgraph_isomorphism_run(2, {{1}, {0}}, 1, {{2}}), std::invalid_argument);
  EXPECT_THROW(a::subgraph_isomorphism_run(-1, {}, 0, {}), std::invalid_argument);
  EXPECT_THROW(a::subgraph_isomorphism_run(1, {}, 0, {}), std::invalid_argument);
  auto r = a::subgraph_isomorphism_run(2, {{1, 1, 0}, {}}, 2, {{}, {0}});
  EXPECT_TRUE(r.match.found);
  a::SubgraphOptions o;
  o.max_matrix_cells = 2;
  EXPECT_THROW(a::subgraph_isomorphism_run(2, {{1}, {0}}, 2, {{1}, {0}}, o), std::length_error);
}
