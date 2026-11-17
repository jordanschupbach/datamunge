#include <gtest/gtest.h>
#include <datamunge/algorithms/polynomial_gf.hpp>
#include <datamunge/algorithms/postman_sort.hpp>
#include <datamunge/algorithms/power_iteration.hpp>
#include <datamunge/algorithms/powerset_examples.hpp>
#include <datamunge/algorithms/pratt_parser.hpp>
#include <datamunge/algorithms/prim.hpp>
#include <datamunge/algorithms/prufer.hpp>
#include <datamunge/algorithms/push_relabel.hpp>
#include <datamunge/algorithms/qft.hpp>
#include <datamunge/algorithms/q_learning.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
namespace {
using U = std::uint64_t;
using W = unsigned __int128;
U residue(std::int64_t x, U p) {
  auto y = x % std::int64_t(p);
  return y < 0 ? U(y + std::int64_t(p)) : U(y);
}
std::vector<U> polynomial(const a::GFPoly& v, U p) {
  std::vector<U> r;
  for (auto x : v)
    r.push_back(residue(x, p));
  while (!r.empty() && !r.back())
    r.pop_back();
  return r;
}
std::vector<U> add_product(const a::GFPoly& q, const a::GFPoly& b, const a::GFPoly& r, U p) {
  std::vector<U> v(std::max(q.empty() || b.empty() ? 0 : q.size() + b.size() - 1, r.size()));
  for (std::size_t i = 0; i < q.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j)
      v[i + j] = U((W(v[i + j]) + W(residue(q[i], p)) * residue(b[j], p)) % p);
  for (std::size_t i = 0; i < r.size(); ++i)
    v[i] = U((W(v[i]) + residue(r[i], p)) % p);
  while (!v.empty() && !v.back())
    v.pop_back();
  return v;
}
long double norm(const std::vector<double>& v) {
  long double s = 0;
  for (auto x : v)
    s += static_cast<long double>(x) * x;
  return std::sqrt(s);
}
double minimum_cut(std::size_t n, const std::vector<a::CapacityEdge>& edges) {
  double best = std::numeric_limits<double>::infinity();
  for (U mask = 0; mask < (U(1) << n); ++mask) {
    if (!(mask & 1) || (mask & (U(1) << (n - 1))))
      continue;
    double c = 0;
    for (auto [u, v, w] : edges)
      if ((mask & (U(1) << u)) && !(mask & (U(1) << v)))
        c += w;
    best = std::min(best, c);
  }
  return best;
}
// Exact exhaustive spanning-tree oracle with a deliberately small Boolean connectivity matrix.
double tree_oracle(std::size_t n, const std::vector<a::WeightedEdge>& edges) {
  double best = std::numeric_limits<double>::infinity();
  for (U mask = 0; mask < (U(1) << edges.size()); ++mask) {
    if (std::popcount(mask) != n - 1)
      continue;
    std::vector<std::vector<bool>> reach(n, std::vector<bool>(n));
    double                         sum = 0;
    for (std::size_t i = 0; i < n; ++i)
      reach[i][i] = true;
    for (std::size_t i = 0; i < edges.size(); ++i)
      if (mask & (U(1) << i)) {
        auto [u, v, w] = edges[i];
        reach[u][v] = reach[v][u] = true;
        sum += w;
      }
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
    if (std::all_of(reach[0].begin(), reach[0].end(), [](bool v) { return v; }))
      best = std::min(best, sum);
  }
  return best;
}
} // namespace
TEST(DivisionActual, ExactIdentityAndEveryRecordedCancellationIncludingWideSignedCoefficients) {
  std::mt19937_64 rng(294);
  for (U p : {U(2), U(7), U(101), U(2305843009213693951)})
    for (int k = 0; k < 80; ++k) {
      a::GFPoly dividend(k % 14), divisor(1 + k % 5);
      for (auto& v : dividend)
        v = std::int64_t(rng() % 100000) - 50000;
      for (auto& v : divisor)
        v = std::int64_t(rng() % 100000) - 50000;
      if (polynomial(divisor, p).empty())
        divisor.back() = 1;
      auto t = a::poly_divmod_gf_trace(dividend, divisor, p);
      EXPECT_EQ(
          add_product(t.result.quotient, divisor, t.result.remainder, p), polynomial(dividend, p));
      EXPECT_TRUE(
          t.result.remainder.empty() || t.result.remainder.size() < polynomial(divisor, p).size());
      EXPECT_EQ(t.coefficient_updates, t.steps.size() * t.divisor.size());
      a::GFPoly before = t.dividend;
      for (const auto& s : t.steps) {
        EXPECT_EQ(s.before, before);
        a::GFPoly term(s.shift + 1, 0);
        term.back() = s.coefficient;
        EXPECT_EQ(add_product(term, t.divisor, s.after, p), polynomial(s.before, p));
        EXPECT_TRUE(s.after.empty() || s.after.size() < s.before.size());
        EXPECT_EQ(residue(s.quotient[s.shift], p), U(s.coefficient));
        before = s.after;
      }
      EXPECT_EQ(before, t.result.remainder);
    }
}
TEST(DivisionActual, ExactHistoryAdmissionAndCompleteFieldGuards) {
  auto        t       = a::poly_divmod_gf_trace({1, 0, 1}, {1, 1}, 7);
  std::size_t entries = t.dividend.size() + t.divisor.size();
  for (auto& s : t.steps)
    entries += 1 + s.before.size() + s.after.size() + s.quotient.size();
  a::PolynomialDivisionOptions o;
  o.max_trace_entries = entries;
  EXPECT_NO_THROW(a::poly_divmod_gf_trace({1, 0, 1}, {1, 1}, 7, o));
  o.max_trace_entries = entries - 1;
  EXPECT_THROW(a::poly_divmod_gf_trace({1, 0, 1}, {1, 1}, 7, o), std::length_error);
  o           = {};
  o.max_steps = 1;
  EXPECT_THROW(a::poly_divmod_gf_trace({1, 0, 1}, {1, 1}, 7, o), std::length_error);
  EXPECT_THROW(a::poly_divmod_gf_trace({1}, {0, 0}, 7), std::invalid_argument);
  EXPECT_THROW(a::poly_divmod_gf_trace({1}, {1}, 9), std::invalid_argument);
}
TEST(PostmanActual, StdSortAndActualNodeHistogramCertificatesAcrossCutoffs) {
  std::mt19937_64 rng(294303);
  for (std::size_t cutoff : {0, 1, 32, 100})
    for (int k = 0; k < 160; ++k) {
      std::vector<std::uint32_t> v(k % 80);
      for (auto& x : v)
        x = std::uint32_t(rng());
      if (k % 3 == 0)
        for (auto& x : v)
          x %= 7;
      auto expected = v;
      std::sort(expected.begin(), expected.end());
      a::PostmanOptions o;
      o.insertion_cutoff = cutoff;
      auto t             = a::postman_sort_trace(v, o);
      EXPECT_EQ(t.run.sorted, expected);
      EXPECT_EQ(
          t.run.operations,
          t.run.key_scans + t.run.distributions + t.run.copybacks + t.run.comparisons +
              t.run.moves + t.run.prefix_steps);
      EXPECT_EQ(t.run.prefix_steps, 256 * t.run.radix_nodes);
      EXPECT_EQ(t.nodes.size(), t.run.nodes);
      for (const auto& node : t.nodes) {
        ASSERT_LE(node.hi, expected.size());
        ASSERT_LT(node.lo, node.hi);
        if (node.insertion)
          continue;
        std::array<std::size_t, 256> counts{};
        for (auto i = node.lo; i < node.hi; ++i)
          ++counts[(expected[i] >> node.shift) & 255];
        EXPECT_EQ(node.counts, counts);
        std::size_t sum = 0;
        for (std::size_t i = 0; i < 256; ++i) {
          EXPECT_EQ(node.offsets[i], sum);
          sum += counts[i];
        }
        EXPECT_EQ(node.offsets[256], sum);
        EXPECT_EQ(sum, node.hi - node.lo);
      }
    }
}
TEST(PostmanActual, FullWordsPolicyAndOriginalGenerator) {
  auto            v = a::postman_uniform_keys(8000, 22);
  std::mt19937_64 g(22);
  for (auto x : v)
    EXPECT_EQ(x, std::uint32_t(g()));
  auto x = std::vector<std::uint32_t>{UINT32_MAX, 0, UINT32_MAX, 1};
  a::postman_sort(x);
  EXPECT_EQ(x, (std::vector<std::uint32_t>{0, 1, UINT32_MAX, UINT32_MAX}));
  a::PostmanOptions o;
  o.max_operations = 1;
  EXPECT_THROW(a::postman_sort_run({3, 2, 1}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  o.insertion_cutoff  = 0;
  EXPECT_THROW(a::postman_sort_trace({2, 1}, o), std::length_error);
  EXPECT_THROW(a::postman_uniform_keys(3, 22, 2), std::length_error);
}
TEST(PowerActual, EveryActualProductNormalizationAndRayleighCertificate) {
  std::vector<std::vector<double>> matrix{{2, 1}, {1, 2}};
  auto                             t = a::power_iteration_trace(matrix, {1, 0}, 15);
  EXPECT_NEAR(t.steps[0].rayleigh_before, 2, 1e-15);
  EXPECT_NEAR(t.steps[1].rayleigh_before, 2.8, 1e-14);
  for (auto& s : t.steps) {
    std::vector<long double> product(2);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        product[i] += static_cast<long double>(matrix[i][j]) * s.before[j];
    long double ray = 0;
    for (int i = 0; i < 2; ++i) {
      EXPECT_NEAR(s.product[i], product[i], 1e-14);
      ray += s.before[i] * product[i];
    }
    EXPECT_NEAR(s.rayleigh_before, ray, 1e-14);
    EXPECT_NEAR(s.product_norm, norm(s.product), 1e-14);
    EXPECT_NEAR(norm(s.after), 1, 1e-14);
    for (int i = 0; i < 2; ++i)
      EXPECT_NEAR(s.after[i], s.product[i] / s.product_norm * (s.sign_flipped ? -1 : 1), 1e-14);
  }
  EXPECT_NEAR(t.run.pair.value, 3, 1e-12);
  EXPECT_LT(t.run.residual_norm, 1e-6);
  EXPECT_EQ(t.run.matvec_calls, 16);
  EXPECT_EQ(t.run.scalar_products, 64);
}
TEST(PowerActual, GapSignZeroAndShapePolicies) {
  auto negative = a::power_iteration_from({{-3, 0}, {0, 1}}, {1, 1}, 30);
  EXPECT_NEAR(negative.pair.value, -3, 1e-12);
  EXPECT_LT(negative.residual_norm, 1e-12);
  auto tied = a::power_iteration_from({{1, 0}, {0, -1}}, {1, 1}, 10);
  EXPECT_NEAR(tied.residual_norm, 1, 1e-14);
  auto zero = a::power_iteration_trace({{0, 0}, {0, 0}}, {1, 0}, 10);
  ASSERT_EQ(zero.steps.size(), 1);
  EXPECT_FALSE(zero.steps[0].accepted);
  EXPECT_TRUE(zero.run.tiny_product);
  EXPECT_EQ(zero.run.accepted, 0);
  EXPECT_THROW(a::power_iteration({{1, 2}, {3}}, 1), std::invalid_argument);
  EXPECT_THROW(a::power_iteration_from({{1}}, {0}, 1), std::invalid_argument);
  EXPECT_THROW(a::power_iteration({{1}}, -1), std::invalid_argument);
  EXPECT_THROW(a::power_iteration({{1e308}}, 1), std::overflow_error);
  a::PowerIterationOptions o;
  o.max_trace_entries = 7;
  EXPECT_THROW(a::power_iteration_trace({{2, 1}, {1, 2}}, {1, 0}, 1, o), std::length_error);
}
TEST(PowersetActual, ExhaustiveSuffixLanguageAndActualSubsetMoves) {
  for (int n = 1; n <= 6; ++n) {
    auto machine = a::nth_from_end_nfa(n);
    auto t       = a::powerset_construction_trace(machine, false, 128);
    EXPECT_EQ(t.dfa.num_states, 1 << n);
    EXPECT_EQ(t.transition_unions, 2 * t.subsets.size());
    for (std::size_t i = 0; i < t.subsets.size(); ++i)
      for (int symbol = 0; symbol < 2; ++symbol) {
        std::set<int> expected;
        for (auto state : t.subsets[i])
          for (auto target : machine.delta[state][symbol])
            expected.insert(target);
        EXPECT_EQ(
            t.subsets[t.dfa.delta[i][symbol]],
            (std::vector<int>(expected.begin(), expected.end())));
      }
    for (int length = 0; length <= 9; ++length)
      for (int mask = 0; mask < (1 << length); ++mask) {
        std::vector<int> word;
        for (int i = 0; i < length; ++i)
          word.push_back((mask >> i) & 1);
        bool expected = length >= n && word[length - n] == 1;
        EXPECT_EQ(a::dfa_accepts(t.dfa, word), expected);
      }
  }
}
TEST(PowersetActual, EmptyPoliciesCompleteValidationAndCaps) {
  a::NFA empty{0, 1, {}, {}, {}};
  EXPECT_EQ(a::powerset_construction_trace(empty, false).dfa.start, -1);
  auto complete = a::powerset_construction_trace(empty, true);
  ASSERT_EQ(complete.subsets.size(), 1);
  EXPECT_TRUE(complete.subsets[0].empty());
  EXPECT_THROW(a::nth_from_end_nfa(0), std::invalid_argument);
  EXPECT_THROW(a::nth_from_end_nfa(4, 3), std::length_error);
  EXPECT_THROW(
      a::powerset_construction_trace(a::nth_from_end_nfa(6), false, 8), std::runtime_error);
}
TEST(PrattActual, ActualOperationOrderPowersAndFullDecimalConsumption) {
  auto              t = a::pratt_trace("1+2*3");
  std::vector<char> ops;
  for (auto& e : t.events) {
    if (e.kind == a::PrattEventKind::Infix)
      ops.push_back(e.op);
    if (e.kind == a::PrattEventKind::Binding)
      EXPECT_EQ(
          e.descend, e.left_binding_power != 0 && e.left_binding_power > e.minimum_binding_power);
  }
  EXPECT_EQ(ops, (std::vector<char>{'*', '+'}));
  EXPECT_EQ(t.run.value, 7);
  EXPECT_EQ(a::pratt_eval("2^3^2"), 512);
  EXPECT_EQ(a::pratt_eval("-2^2"), 4);
  EXPECT_EQ(a::pratt_eval("-(2^2)"), -4);
  EXPECT_EQ(a::pratt_eval(" .5 + 5. "), 5.5);
  for (auto s : {"1.2.3", ".", "1e3", "1+", "(2", "2)"})
    EXPECT_THROW(a::pratt_eval(s), std::runtime_error);
  EXPECT_THROW(a::pratt_eval(std::string("1\0+2", 4)), std::runtime_error);
  EXPECT_THROW(a::pratt_eval("1/0"), std::overflow_error);
}
TEST(PrattActual, RecursivePoliciesAndFiniteArithmetic) {
  a::PrattOptions o;
  o.max_depth = 8;
  EXPECT_THROW(a::pratt_run("((((1))))", o), std::length_error);
  o                = {};
  o.max_operations = 2;
  EXPECT_THROW(a::pratt_run("1+2", o), std::length_error);
  o                   = {};
  o.max_trace_records = 1;
  EXPECT_THROW(a::pratt_trace("1", o), std::length_error);
  EXPECT_EQ(a::pratt_run("1+2").bytes_consumed, 3);
  EXPECT_THROW(a::pratt_eval("(-1)^.5"), std::overflow_error);
}
TEST(PrimActual, ExhaustiveSmallTreeOracleAndActualHeapCertificates) {
  std::mt19937_64 rng(301);
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<a::WeightedEdge> edges;
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j)
        edges.emplace_back(i, j, double(int(rng() % 19) - 9));
    auto t = a::prim_trace(4, edges);
    EXPECT_EQ(t.run.tree.total_weight, tree_oracle(4, edges));
    std::multiset<std::tuple<double, std::size_t, std::size_t>> frontier;
    std::set<std::size_t>                                       joined{0};
    std::size_t                                                 accepted = 0, stale = 0;
    for (auto& e : t.events) {
      auto key = std::tuple{e.weight, e.from, e.to};
      if (e.push) {
        EXPECT_TRUE(joined.contains(e.from));
        frontier.insert(key);
      } else {
        ASSERT_FALSE(frontier.empty());
        EXPECT_EQ(*frontier.begin(), key);
        frontier.erase(frontier.begin());
        EXPECT_EQ(e.accepted, !joined.contains(e.to));
        if (e.accepted) {
          joined.insert(e.to);
          ++accepted;
        } else
          ++stale;
      }
      EXPECT_EQ(e.frontier_after, frontier.size());
      EXPECT_EQ(e.reached_after, joined.size());
    }
    EXPECT_EQ(accepted, 3);
    EXPECT_EQ(stale, t.run.stale);
    EXPECT_EQ(t.run.operations, t.run.adjacency_entries + t.run.pushes + t.run.pops);
  }
}
TEST(PrimActual, DisconnectedFiniteMultigraphAndPolicies) {
  auto r = a::prim_run(4, {{0, 0, -9}, {0, 1, 5}, {0, 1, 1}});
  EXPECT_EQ(r.tree.total_weight, 1);
  EXPECT_EQ(r.reached, 2);
  EXPECT_FALSE(r.tree.is_connected);
  EXPECT_THROW(
      a::prim(3, {{1, 2, std::numeric_limits<double>::infinity()}}, 0), std::invalid_argument);
  a::PrimOptions o;
  o.max_trace_records = 1;
  EXPECT_THROW(a::prim_trace(2, {{0, 1, 1}}, 0, o), std::length_error);
  EXPECT_THROW(a::prim(0, {}), std::invalid_argument);
}
TEST(PruferActual, AllSmallCodesDegreeLawRoundTripAndRealLeafSets) {
  for (std::size_t n = 2; n <= 6; ++n) {
    std::size_t count = 1;
    for (std::size_t i = 0; i < n - 2; ++i)
      count *= n;
    std::vector<std::size_t> code(n - 2);
    for (std::size_t index = 0; index < count; ++index) {
      auto value = index;
      for (auto& digit : code) {
        digit = value % n;
        value /= n;
      }
      auto t = a::prufer_to_tree_trace(code);
      ASSERT_EQ(t.edges.size(), n - 1);
      std::vector<std::size_t> degree(n), expected(n, 1);
      for (auto [u, v] : t.edges) {
        ASSERT_NE(u, v);
        ++degree[u];
        ++degree[v];
      }
      for (auto digit : code)
        ++expected[digit];
      EXPECT_EQ(degree, expected);
      auto e = a::tree_to_prufer_trace(n, t.edges);
      EXPECT_EQ(e.sequence, code);
      for (auto& s : e.steps) {
        ASSERT_FALSE(s.leaves_before.empty());
        EXPECT_EQ(s.leaf, s.leaves_before.front());
        EXPECT_EQ(s.neighbor_degree_after + 1, s.neighbor_degree_before);
        EXPECT_TRUE(std::is_sorted(s.leaves_after.begin(), s.leaves_after.end()));
      }
      EXPECT_TRUE(t.steps.back().final_edge);
      EXPECT_TRUE(t.steps.back().leaves_after.empty());
    }
  }
}
TEST(PruferActual, NonTreeGuardsAndExactEnumeration) {
  EXPECT_THROW(a::tree_to_prufer(4, {{0, 1}, {1, 2}, {2, 0}}), std::invalid_argument);
  EXPECT_THROW(a::tree_to_prufer(3, {{0, 1}, {0, 1}}), std::invalid_argument);
  EXPECT_THROW(a::tree_to_prufer(2, {{0, 0}}), std::invalid_argument);
  EXPECT_THROW(a::prufer_to_tree({3}), std::invalid_argument);
  auto r = a::prufer_enumeration(4);
  EXPECT_EQ(r.sequences, 16);
  EXPECT_EQ(r.distinct_trees, 16);
  EXPECT_EQ(r.cayley_count, 16);
  EXPECT_THROW(a::prufer_enumeration(5, 100), std::length_error);
  a::PruferOptions o;
  o.max_trace_entries = 2;
  EXPECT_THROW(a::prufer_to_tree_trace({}, o), std::length_error);
}
TEST(PreflowActual, ExhaustiveCutOracleOnIntegerMultigraphs) {
  std::mt19937_64 rng(302);
  for (std::size_t n = 2; n <= 6; ++n)
    for (int trial = 0; trial < 60; ++trial) {
      std::vector<a::CapacityEdge> edges;
      for (std::size_t i = 0; i < n * 2; ++i)
        edges.emplace_back(rng() % n, rng() % n, double(rng() % 10));
      auto t = a::push_relabel_trace(n, edges, 0, n - 1);
      EXPECT_EQ(t.run.result.max_flow, minimum_cut(n, edges));
      auto cert = a::max_flow_certificate(n, edges, 0, n - 1, t.run.result);
      EXPECT_TRUE(cert.feasible);
      EXPECT_TRUE(cert.optimal_within_tolerance);
      for (std::size_t v = 1; v + 1 < n; ++v)
        EXPECT_EQ(t.run.excess[v], 0);
      EXPECT_EQ(
          t.run.work.operations,
          t.run.work.arc_inspections + t.run.work.pushes + t.run.work.relabels +
              t.run.work.queue_pops);
      std::vector<double>      excess(n), flows(edges.size());
      std::vector<std::size_t> height(n);
      height[0] = n;
      for (auto& e : t.events) {
        EXPECT_EQ(height[e.from], e.height_before);
        if (e.kind == a::PreflowEventKind::Relabel) {
          EXPECT_GT(e.height_after, e.height_before);
          height[e.from] = e.height_after;
        } else {
          EXPECT_EQ(excess[e.from], e.from_excess_before);
          EXPECT_EQ(excess[e.to], e.to_excess_before);
          EXPECT_GT(e.amount, 0);
          EXPECT_EQ(e.residual_after, e.residual_before - e.amount);
          EXPECT_EQ(e.reverse_after, e.reverse_before + e.amount);
          EXPECT_EQ(e.from_excess_after, e.from_excess_before - e.amount);
          EXPECT_EQ(e.to_excess_after, e.to_excess_before + e.amount);
          if (!e.preflow)
            EXPECT_EQ(height[e.from], height[e.to] + 1);
          excess[e.from] = e.from_excess_after;
          excess[e.to]   = e.to_excess_after;
          flows[e.input] += e.forward ? e.amount : -e.amount;
        }
      }
      EXPECT_EQ(excess, t.run.excess);
      EXPECT_EQ(flows, t.run.result.edge_flow);
    }
}
TEST(PreflowActual, TinyCapacitySelfLoopProgressAndPolicies) {
  auto tiny = a::push_relabel_max_flow(3, {{0, 1, 1e-15}, {1, 2, 1e-15}}, 0, 2);
  EXPECT_DOUBLE_EQ(tiny.max_flow, 1e-15);
  auto loop = a::push_relabel_max_flow(2, {{0, 0, 5}, {0, 1, 3}, {1, 1, 7}}, 0, 1);
  EXPECT_EQ(loop.edge_flow, (std::vector<double>{0, 3, 0}));
  a::PushRelabelOptions o;
  o.max_operations = 1;
  EXPECT_THROW(a::push_relabel_run(2, {{0, 1, 3}}, 0, 1, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 6;
  EXPECT_THROW(a::push_relabel_trace(2, {{0, 1, 3}}, 0, 1, o), std::length_error);
  EXPECT_THROW(a::push_relabel_max_flow(2, {{0, 1, -1}}, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      a::push_relabel_max_flow(2, {{0, 1, 1e300}, {0, 1, 1e-10}}, 0, 1), std::runtime_error);
}
TEST(FourierActual, IndependentLongDoubleDFTForArbitraryInputsAndInverse) {
  std::mt19937_64 rng(303);
  for (int bits = 0; bits <= 6; ++bits) {
    std::size_t     N = std::size_t{1} << bits;
    a::QuantumState input(N);
    for (auto& z : input)
      z = {double(int(rng() % 101) - 50) / 50, double(int(rng() % 101) - 50) / 50};
    for (bool inverse : {false, true}) {
      auto t = a::qft_trace(input, bits, inverse);
      for (std::size_t k = 0; k < N; ++k) {
        std::complex<long double> expected{};
        for (std::size_t j = 0; j < N; ++j) {
          auto theta = (inverse ? -1 : 1) * 2 * std::numbers::pi_v<long double> * j * k / N;
          expected += std::complex<long double>{input[j].real(), input[j].imag()} *
                      std::exp(std::complex<long double>{0, theta});
        }
        expected /= std::sqrt(static_cast<long double>(N));
        EXPECT_NEAR(t.run.state[k].real(), expected.real(), 3e-12);
        EXPECT_NEAR(t.run.state[k].imag(), expected.imag(), 3e-12);
      }
      EXPECT_EQ(t.run.hadamards, bits);
      EXPECT_EQ(t.run.phase_gates, bits * (bits - 1) / 2);
      EXPECT_EQ(t.run.swaps, bits / 2);
      long double initial_norm = 0;
      for (auto z : input)
        initial_norm += std::norm(z);
      for (auto& e : t.gates) {
        long double after = 0;
        for (auto z : e.after)
          after += std::norm(z);
        EXPECT_NEAR(after, initial_norm, 2e-11);
      }
    }
    auto forward = a::qft_run(input, bits);
    auto reverse = a::qft_run(forward.state, bits, true);
    for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(std::abs(reverse.state[i] - input[i]), 0, 3e-12);
  }
}
TEST(FourierActual, CombShapesFiniteGuardsAndExactHistoryPolicy) {
  auto s = a::qft_comb(4, 4, 1);
  for (std::size_t i = 0; i < 16; ++i)
    EXPECT_EQ(s[i].real(), i % 4 == 1 ? .5 : 0);
  auto          t     = a::qft_trace(a::qstate_zero(2), 2);
  std::size_t   exact = 8 + t.gates.size() * 5;
  a::QFTOptions o;
  o.max_trace_entries = exact;
  EXPECT_NO_THROW(a::qft_trace(a::qstate_zero(2), 2, false, o));
  o.max_trace_entries = exact - 1;
  EXPECT_THROW(a::qft_trace(a::qstate_zero(2), 2, false, o), std::length_error);
  a::QuantumState malformed(3);
  auto            saved = malformed;
  EXPECT_THROW(a::qft(malformed, 2), std::invalid_argument);
  EXPECT_EQ(malformed, saved);
  EXPECT_THROW(a::qft_comb(2, 0), std::invalid_argument);
  EXPECT_THROW(a::qft_comb(2, 1, 4), std::invalid_argument);
  EXPECT_THROW(
      a::qft_run({{std::numeric_limits<double>::infinity(), 0}}, 0), std::invalid_argument);
}
TEST(QLearningActual, TerminalAndNonterminalTargetsHaveIndependentClosedForms) {
  a::TabularEnvironment env{
      1,
      1,
      [] { return std::size_t{0}; },
      [](std::size_t, std::size_t) { return a::StepResult{0, 2, true}; }};
  a::RLParameters p;
  p.learning_rate = .5;
  p.discount      = 1;
  p.epsilon       = 0;
  p.episodes      = 2;
  auto terminal   = a::q_learning_trace(env, p);
  ASSERT_EQ(terminal.steps.size(), 2);
  EXPECT_EQ(terminal.steps[0].target, 2);
  EXPECT_EQ(terminal.steps[1].target, 2);
  EXPECT_EQ(terminal.run.result.q[0][0], 1.5);
  env.step    = [](std::size_t, std::size_t) { return a::StepResult{0, 1, false}; };
  p.discount  = .5;
  p.max_steps = 2;
  auto t      = a::q_learning_trace(env, p);
  ASSERT_EQ(t.steps.size(), 4);
  for (std::size_t i = 0; i < t.steps.size(); ++i) {
    EXPECT_DOUBLE_EQ(t.steps[i].after, 2 * (1 - std::pow(.75, double(i + 1))));
    EXPECT_DOUBLE_EQ(t.steps[i].target, 1 + .5 * t.steps[i].before);
    EXPECT_FALSE(t.steps[i].explored);
  }
  EXPECT_EQ(t.run.transitions, 4);
  EXPECT_EQ(t.run.reset_calls, 2);
  EXPECT_EQ(t.run.greedy_actions, 4);
  EXPECT_EQ(t.run.terminated, (std::vector<bool>{false, false}));
}
TEST(QLearningActual, CompleteGuardsBeforeCallbacksAndTransitionCaps) {
  std::size_t           reset = 0, step = 0;
  a::TabularEnvironment env{
      1,
      1,
      [&] {
        ++reset;
        return std::size_t{0};
      },
      [&](std::size_t, std::size_t) {
        ++step;
        return a::StepResult{0, 1, false};
      }};
  a::RLParameters p;
  p.episodes        = 1;
  p.max_steps       = 2;
  p.epsilon         = 0;
  auto bad          = p;
  bad.learning_rate = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(a::q_learning(env, bad), std::invalid_argument);
  EXPECT_EQ(reset, 0);
  a::QLearningOptions o;
  o.max_transitions = 1;
  EXPECT_THROW(a::q_learning_run(env, p, o), std::length_error);
  EXPECT_EQ(step, 1);
  env.step = [](std::size_t, std::size_t) { return a::StepResult{1, 0, true}; };
  EXPECT_THROW(a::q_learning(env, p), std::invalid_argument);
  env.step = [](std::size_t, std::size_t) {
    return a::StepResult{0, std::numeric_limits<double>::infinity(), true};
  };
  EXPECT_THROW(a::q_learning(env, p), std::invalid_argument);
  p.episodes = 0;
  auto empty = a::q_learning_run(a::tabular_corridor(2), p);
  EXPECT_EQ(empty.transitions, 0);
  EXPECT_TRUE(empty.result.episode_returns.empty());
  p.episodes  = 3;
  p.max_steps = 0;
  auto zero   = a::q_learning_run(a::tabular_corridor(2), p);
  EXPECT_EQ(zero.reset_calls, 3);
  EXPECT_EQ(zero.transitions, 0);
  EXPECT_EQ(zero.terminated, (std::vector<bool>{false, false, false}));
}
TEST(QLearningActual, NativeFixturesActualRolloutsAndDescriptiveSmoothing) {
  auto env = a::tabular_grid(4);
  EXPECT_EQ(env.step(0, 0).next_state, 0);
  EXPECT_EQ(env.step(0, 1).next_state, 1);
  EXPECT_EQ(env.step(14, 1).next_state, 15);
  EXPECT_TRUE(env.step(14, 1).terminal);
  a::RLParameters p;
  p.learning_rate = .5;
  p.discount      = .95;
  p.epsilon       = .1;
  p.episodes      = 500;
  p.seed          = 42;
  auto trained    = a::q_learning_run(env, p);
  auto rollout    = a::tabular_policy_rollout(env, trained.result.greedy_policy, 100);
  EXPECT_TRUE(rollout.terminal);
  EXPECT_EQ(rollout.actions.size(), 6);
  EXPECT_EQ(rollout.total_reward, -6);
  EXPECT_EQ(a::trailing_return_mean({1, 3, 8}, 2), (std::vector<double>{1, 2, 5.5}));
  EXPECT_THROW(a::trailing_return_mean({1}, 0), std::invalid_argument);
  EXPECT_THROW(a::tabular_grid(1), std::invalid_argument);
  EXPECT_THROW(a::tabular_policy_rollout(a::tabular_corridor(2), {2, 1}), std::invalid_argument);
}
