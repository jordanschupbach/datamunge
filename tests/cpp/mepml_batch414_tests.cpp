#include <gtest/gtest.h>

#include <datamunge/algorithms/unary_coding.hpp>
#include <datamunge/algorithms/uniform_binary_search.hpp>
#include <datamunge/algorithms/uniform_cost_search.hpp>
#include <datamunge/algorithms/union_merge_analysis.hpp>
#include <datamunge/algorithms/upgma.hpp>
#include <datamunge/algorithms/vector_clock.hpp>
#include <datamunge/algorithms/velocity_verlet.hpp>
#include <datamunge/algorithms/velvet.hpp>
#include <datamunge/algorithms/verhoeff.hpp>
#include <datamunge/algorithms/vincenty.hpp>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
TEST(Batch414, UnaryRoundTrip) {
  std::vector<std::uint64_t> v{0, 1, 2, 3, 5, 8};
  auto                       b = a::unary_encode(v);
  EXPECT_EQ(b.size(), 25);
  EXPECT_EQ(a::unary_decode(b, 6), v);
  EXPECT_EQ(a::unary_decode({}, 0), std::vector<std::uint64_t>{});
}
TEST(Batch414, UnaryValidation) {
  EXPECT_THROW(a::unary_decode({true, true}, 1), std::invalid_argument);
  EXPECT_THROW(a::unary_decode({false}, 0), std::invalid_argument);
  EXPECT_THROW(a::unary_decode({}, 1), std::invalid_argument);
  EXPECT_THROW(a::unary_decode({}, -1), std::invalid_argument);
  std::vector<bool> bits{true};
  auto              copy = bits;
  EXPECT_THROW(a::unary_encode_value(UINT64_MAX, bits), std::length_error);
  EXPECT_EQ(bits, copy);
}
TEST(Batch414, UniformProbe) {
  std::vector<int> keys(100);
  for (int i = 0; i < 100; ++i)
    keys[i] = i * 3;
  auto r = a::uniform_binary_search_work(keys, 231, a::uniform_search_table(100));
  EXPECT_EQ(r.index, 77);
  EXPECT_EQ(r.probes.size(), 7);
  EXPECT_EQ(r.probes.back().after, 77);
  EXPECT_EQ(a::uniform_search_table(INT_MAX).front(), 1 << 30);
  EXPECT_EQ(a::uniform_search_table(INT_MAX).size(), 31);
}
TEST(Batch414, UniformValidation) {
  EXPECT_EQ(a::uniform_binary_search(std::vector<int>{1, 2, 2}, 2, a::uniform_search_table(3)), 2);
  EXPECT_THROW(
      a::uniform_binary_search(std::vector<int>{2, 1}, 1, a::uniform_search_table(2)),
      std::invalid_argument);
  EXPECT_THROW(a::uniform_binary_search(std::vector<int>{1}, 1, {-1}), std::invalid_argument);
  EXPECT_EQ(a::uniform_binary_search(std::vector<int>{}, 1, {}), std::size_t(-1));
}
TEST(Batch414, UcsActualSettlements) {
  auto graph = a::uniform_cost_example_graph();
  auto r     = a::uniform_cost_search(10, graph, 0, 5);
  EXPECT_TRUE(r.found);
  EXPECT_EQ(r.cost, 6);
  EXPECT_EQ(r.path, (std::vector<int>{0, 1, 3, 5}));
  EXPECT_EQ(r.settled, (std::vector<int>{0, 1, 2, 3, 5}));
  EXPECT_EQ(r.expanded, 5);
}
TEST(Batch414, UcsValidationAndLargeCost) {
  auto r = a::uniform_cost_search(2, {{{1, 1.1e300}}, {}}, 0, 1);
  EXPECT_TRUE(r.found);
  EXPECT_DOUBLE_EQ(r.cost, 1.1e300);
  EXPECT_THROW(a::uniform_cost_search(2, {{}, {{0, -1}}}, 0, 0), std::invalid_argument);
  EXPECT_THROW(a::uniform_cost_search(2, {{{2, 1}}, {}}, 0, 1), std::invalid_argument);
  EXPECT_FALSE(a::uniform_cost_search(2, {{}, {}}, 0, 1).found);
  EXPECT_TRUE(a::uniform_cost_search(1, {{}}, 0, 0).found);
}
TEST(Batch414, UnionSameProducer) {
  auto r = a::k_way_merge_trace({{1, 1}, {1, 2}}, true);
  EXPECT_EQ(r.values, (std::vector<int>{1, 2}));
  EXPECT_EQ(r.work.pops, 4);
  EXPECT_EQ(r.work.discarded_duplicates, 2);
  std::size_t emitted = 0;
  for (auto q : r.steps)
    emitted += q.emitted;
  EXPECT_EQ(emitted, r.values.size());
}
TEST(Batch414, UnionFixture) {
  for (int range : {1, 20, 200}) {
    auto          lists = a::union_overlap_lists(range);
    std::set<int> oracle;
    for (auto row : lists) {
      EXPECT_TRUE(std::is_sorted(row.begin(), row.end()));
      oracle.insert(row.begin(), row.end());
    }
    EXPECT_EQ(a::union_merge(lists), (std::vector<int>(oracle.begin(), oracle.end())));
    EXPECT_EQ(a::k_way_merge_analyze(lists, true).work.pops, 500);
  }
  EXPECT_THROW(a::union_overlap_lists(0), std::invalid_argument);
}
TEST(Batch414, UpgmaActualMeans) {
  auto r = a::upgma({{0, 2, 6}, {2, 0, 4}, {6, 4, 0}});
  ASSERT_EQ(r.merges.size(), 2);
  EXPECT_EQ(r.merges[0].height, 1);
  EXPECT_EQ(r.merges[1].height, 2.5);
  EXPECT_EQ(r.merges[1].right_size, 2);
  EXPECT_EQ(r.pair_checks, 4);
  EXPECT_EQ(r.distance_updates, 1);
  EXPECT_FALSE(a::upgma_plot(r, 3).series().empty());
}
TEST(Batch414, UpgmaValidation) {
  EXPECT_TRUE(a::upgma({}).merges.empty());
  EXPECT_TRUE(a::upgma({{0}}).merges.empty());
  EXPECT_THROW(a::upgma({{0, 1}, {2, 0}}), std::invalid_argument);
  EXPECT_THROW(a::upgma({{0, 1}, {1}}), std::invalid_argument);
  double m = std::numeric_limits<double>::max();
  auto   r = a::upgma({{0, m, m}, {m, 0, m}, {m, m, 0}});
  EXPECT_TRUE(std::isfinite(r.merges.back().height));
}
TEST(Batch414, VectorProtocol) {
  a::VectorClock c0(3), c1(3), c2(3);
  a::vc_event(c0, 0);
  auto A = c0;
  auto S = a::vc_send(c0, 0);
  a::vc_receive(c1, 1, S);
  a::vc_event(c2, 2);
  EXPECT_EQ(c1.v, (std::vector<int>{2, 1, 0}));
  EXPECT_TRUE(a::vc_happens_before(A, c1));
  EXPECT_TRUE(a::vc_concurrent(c1, c2));
  EXPECT_FALSE(a::vc_concurrent(c1, c1));
}
TEST(Batch414, VectorValidationAtomic) {
  a::VectorClock c(2), m(2);
  c.v         = {1, INT_MAX};
  m.v         = {9, 0};
  auto before = c.v;
  EXPECT_THROW(a::vc_receive(c, 1, m), std::overflow_error);
  EXPECT_EQ(c.v, before);
  EXPECT_THROW(a::vc_event(c, 2), std::invalid_argument);
  a::VectorClock shortc(1);
  EXPECT_THROW(a::vc_concurrent(c, shortc), std::invalid_argument);
  c.v[0] = -1;
  EXPECT_THROW(a::vc_event(c, 0), std::invalid_argument);
}
TEST(Batch414, VelvetCompleteWalk) {
  auto r = a::velvet_work({"AT", "TC", "CG"}, 2);
  EXPECT_TRUE(r.eulerian);
  EXPECT_EQ(r.sequence, "ATCG");
  EXPECT_EQ(r.consumed, 3);
  EXPECT_EQ(r.walk.size(), 4);
  EXPECT_EQ(a::velvet_assemble({"AA", "AA"}, 2), "AAA");
}
TEST(Batch414, VelvetRejectsPartial) {
  EXPECT_FALSE(a::velvet_work({"AT", "GC"}, 2).eulerian);
  EXPECT_FALSE(a::velvet_work({"AT", "AG"}, 2).eulerian);
  EXPECT_FALSE(a::velvet_work({"AA", "CC"}, 2).eulerian);
  EXPECT_TRUE(a::velvet_work({}, 2).eulerian);
  EXPECT_THROW(a::velvet_work({"A"}, 1), std::invalid_argument);
}
TEST(Batch414, VerhoeffActualTrace) {
  auto r = a::verhoeff_trace("236");
  EXPECT_EQ(r.check_digit, 3);
  EXPECT_EQ(r.steps.size(), 3);
  EXPECT_EQ(r.steps.front().digit, 6);
  EXPECT_TRUE(a::verhoeff_validate("2363"));
  auto d = a::verhoeff_detection("1230904560907");
  EXPECT_EQ(d.substitutions, d.substitutions_caught);
  EXPECT_EQ(d.transpositions, d.transpositions_caught);
}
TEST(Batch414, VerhoeffValidation) {
  EXPECT_FALSE(a::verhoeff_validate(""));
  EXPECT_FALSE(a::verhoeff_validate("12x3"));
  EXPECT_THROW(a::verhoeff_check_digit("12x"), std::invalid_argument);
  EXPECT_EQ(a::verhoeff_check_digit(""), 0);
  EXPECT_TRUE(a::verhoeff_validate("0"));
  for (int i = 0; i < 100; ++i) {
    auto payload = std::to_string(i);
    auto d       = a::verhoeff_detection(payload);
    EXPECT_EQ(d.substitutions, d.substitutions_caught);
    EXPECT_EQ(d.transpositions, d.transpositions_caught);
  }
}
TEST(Batch414, VerletOperands) {
  auto r = a::velocity_verlet_work(
      [](const std::vector<double>& x) { return std::vector<double>{-x[0]}; },
      {1},
      {0},
      0,
      .1,
      1,
      1);
  ASSERT_EQ(r.steps.size(), 1);
  EXPECT_NEAR(r.steps[0].after_x[0], .995, 1e-15);
  EXPECT_NEAR(r.steps[0].after_v[0], -.09975, 1e-15);
  EXPECT_EQ(r.acceleration_calls, 2);
  auto back = a::velocity_verlet(
      [](const std::vector<double>& x) { return std::vector<double>{-x[0]}; },
      r.solution.x.back(),
      r.solution.v.back(),
      .1,
      0,
      1);
  EXPECT_NEAR(back.x.back()[0], 1, 1e-14);
  EXPECT_NEAR(back.v.back()[0], 0, 1e-14);
}
TEST(Batch414, VerletValidationAndInvariant) {
  EXPECT_THROW(
      a::velocity_verlet(
          [](const std::vector<double>&) { return std::vector<double>{}; }, {1}, {0}, 0, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      a::velocity_verlet(
          [](const std::vector<double>&) { return std::vector<double>{0}; }, {1}, {}, 0, 1, 1),
      std::invalid_argument);
  auto r = a::velocity_verlet(
      [](const std::vector<double>& x) { return std::vector<double>{-x[0]}; },
      {1},
      {0},
      0,
      10,
      100);
  for (std::size_t i = 0; i < r.t.size(); ++i)
    EXPECT_NEAR(
        r.v[i][0] * r.v[i][0] + (1 - .1 * .1 / 4) * r.x[i][0] * r.x[i][0], 1 - .1 * .1 / 4, 1e-13);
}
TEST(Batch414, VincentyEquatorAndStatus) {
  auto r = a::vincenty_work(0, 0, 0, 1, 200, 1e-12, 10);
  EXPECT_TRUE(r.result.converged);
  EXPECT_NEAR(r.result.distance_m, 6378137. * std::numbers::pi / 180, 1e-6);
  EXPECT_EQ(r.steps.size(), r.result.iterations);
  auto fail = a::vincenty_inverse(0, 0, 0, 180);
  EXPECT_FALSE(fail.converged);
  EXPECT_TRUE(std::isnan(fail.distance_m));
  EXPECT_EQ(fail.iterations, 200);
}
TEST(Batch414, VincentyValidationAndPoles) {
  EXPECT_THROW(a::vincenty_inverse(91, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(a::vincenty_inverse(0, 0, 0, 1, 0), std::invalid_argument);
  EXPECT_EQ(a::vincenty_inverse(90, 0, 90, 180).distance_m, 0);
  EXPECT_EQ(a::vincenty_inverse(0, -180, 0, 180).distance_m, 0);
  EXPECT_NEAR(a::haversine_distance(0, 0, 0, 180), 6371000 * std::numbers::pi, 1e-6);
}
