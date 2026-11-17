#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/viterbi.hpp>
#include <datamunge/algorithms/wang_landau.hpp>
#include <datamunge/geometry/clipped_voronoi.hpp>
#include <datamunge/stats/v_statistic.hpp>
#include <limits>
#include <numeric>
namespace a = datamunge::algorithms;
namespace s = datamunge::stats;
namespace g = datamunge::geometry;
TEST(Batch424, ViterbiActualTables) {
  auto r = a::viterbi_work(a::forward_backward_example_model(), {0, 1, 2});
  EXPECT_EQ(r.result.states, (std::vector<std::size_t>{0, 0, 1}));
  EXPECT_EQ(r.transition_terms, 8);
  EXPECT_NEAR(r.result.log_probability, std::log(.6 * .5 * .7 * .4 * .3 * .6), 1e-14);
  EXPECT_EQ(r.predecessor[2][1], 0);
  EXPECT_TRUE(r.result.possible);
}
TEST(Batch424, ViterbiImpossibleAndValidation) {
  auto h     = a::forward_backward_example_model();
  h.emission = {{1, 0, 0}, {1, 0, 0}};
  auto r     = a::viterbi(h, {2});
  EXPECT_FALSE(r.possible);
  EXPECT_TRUE(r.states.empty());
  EXPECT_EQ(r.log_probability, -std::numeric_limits<double>::infinity());
  EXPECT_TRUE(a::viterbi(h, {}).possible);
  h.transition[0].pop_back();
  EXPECT_THROW(a::viterbi(h, {0}), std::invalid_argument);
  h = a::forward_backward_example_model();
  EXPECT_THROW(a::viterbi(h, {3}), std::out_of_range);
}
TEST(Batch424, VoronoiActualBisector) {
  auto r = g::clipped_voronoi({{2, 5}, {8, 5}}, {0, 10, 0, 10}, 2);
  ASSERT_EQ(r.cells.size(), 2);
  EXPECT_EQ(r.halfplanes, 2);
  EXPECT_EQ(r.steps.size(), 2);
  double total_area = 0;
  for (std::size_t i = 0; i < 2; ++i) {
    auto&  cell = r.cells[i];
    double area = 0;
    for (std::size_t j = 0; j < cell.size(); ++j) {
      auto p = cell[j], q = cell[(j + 1) % cell.size()];
      area += p.x * q.y - q.x * p.y;
      EXPECT_LE(i == 0 ? p.x - 5 : 5 - p.x, 1e-12);
    }
    EXPECT_NEAR(std::abs(area) / 2, 50, 1e-12);
    total_area += std::abs(area) / 2;
  }
  EXPECT_EQ(total_area, 100);
  EXPECT_FALSE(g::clipped_voronoi_plot(r).series().empty());
}
TEST(Batch424, VoronoiDegeneracyAndValidation) {
  EXPECT_TRUE(g::clipped_voronoi({}, {0, 1, 0, 1}).cells.empty());
  EXPECT_EQ(g::clipped_voronoi({{.5, .5}}, {0, 1, 0, 1}).cells[0].size(), 4);
  EXPECT_EQ(g::clipped_voronoi({{.1, .5}, {.5, .5}, {.9, .5}}, {0, 1, 0, 1}).cells.size(), 3);
  EXPECT_THROW(g::clipped_voronoi({{0, 0}, {0, 0}}, {0, 1, 0, 1}), std::invalid_argument);
  EXPECT_THROW(g::clipped_voronoi({{0, 0}}, {1, 0, 0, 1}), std::invalid_argument);
  EXPECT_EQ(g::voronoi_example_sites().size(), 24);
}
TEST(Batch424, VStatisticActualTerms) {
  auto t = s::VStatistic(2).trace({1, 3}, s::empirical_variance_kernel, 4);
  EXPECT_TRUE(t.result.exact);
  EXPECT_EQ(t.result.num_terms, 4);
  EXPECT_EQ(t.result.estimate, 1);
  ASSERT_EQ(t.terms.size(), 4);
  EXPECT_EQ(t.terms[0].indices, (std::vector<std::size_t>{0, 0}));
  EXPECT_EQ(t.terms[1].indices, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(t.terms[2].indices, (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(t.terms[3].value, 0);
  auto mc = s::VStatistic(2, {100, 42});
  EXPECT_FALSE(mc.run({1, 3}, s::empirical_variance_kernel).exact);
  EXPECT_EQ(
      mc.run({1, 3}, s::empirical_variance_kernel).estimate,
      mc.trace({1, 3}, s::empirical_variance_kernel).result.estimate);
}
TEST(Batch424, VStatisticValidation) {
  EXPECT_THROW(s::VStatistic(0), std::invalid_argument);
  EXPECT_THROW(s::VStatistic(65), std::invalid_argument);
  EXPECT_THROW(s::VStatistic(2).run({}, s::empirical_variance_kernel), std::invalid_argument);
  EXPECT_THROW(
      s::VStatistic(2).run(
          {1, std::numeric_limits<double>::infinity()}, s::empirical_variance_kernel),
      std::invalid_argument);
  EXPECT_THROW(
      s::VStatistic(2).run(
          {1, 2}, [](const auto&) { return std::numeric_limits<double>::quiet_NaN(); }),
      std::invalid_argument);
  auto r = s::VStatistic(64).run({1, 2}, [](const auto&) { return 1.; });
  EXPECT_FALSE(r.exact);
  EXPECT_EQ(r.num_terms, 200000);
  EXPECT_EQ(r.estimate, 1);
}
TEST(Batch424, WangActualMovesAndChecks) {
  a::WangLandauOptions o;
  o.max_moves         = 120;
  o.check_interval    = 6;
  o.max_move_records  = 120;
  o.max_check_records = 100;
  auto r              = a::wang_landau_work(2, o);
  EXPECT_EQ(r.moves, 120);
  EXPECT_EQ(r.move_records.size(), 120);
  EXPECT_EQ(r.checks, 20);
  std::size_t accepted = 0;
  for (auto q : r.move_records) {
    EXPECT_EQ(std::abs(q.proposed - q.before), 1);
    EXPECT_EQ(q.after, q.accepted ? q.proposed : q.before);
    EXPECT_EQ(q.random_used, q.log_ratio < 0);
    accepted += q.accepted;
  }
  EXPECT_EQ(r.accepted, accepted);
  for (auto q : r.check_records) {
    EXPECT_EQ(q.moves % 6, 0);
    EXPECT_EQ(q.minimum, *std::min_element(q.histogram.begin(), q.histogram.end()));
    EXPECT_EQ(q.flat, q.minimum > 0 && q.minimum >= o.flatness * q.mean);
  }
  EXPECT_EQ(r.log_density[0], 0);
}
TEST(Batch424, WangStatusAndValidation) {
  EXPECT_THROW(a::wang_landau_work(0), std::invalid_argument);
  a::WangLandauOptions o;
  o.flatness = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(a::wang_landau_work(2, o), std::invalid_argument);
  o           = {};
  o.max_moves = 0;
  auto none   = a::wang_landau_work(2, o);
  EXPECT_FALSE(none.complete);
  EXPECT_EQ(none.moves, 0);
  o           = {};
  o.log_f_min = .25;
  auto done   = a::wang_landau_work(2, o);
  EXPECT_TRUE(done.complete);
  EXPECT_EQ(done.completed_stages, 2);
  EXPECT_EQ(done.log_f_remaining, .25);
  EXPECT_EQ(a::wang_landau_dos(2, .85, .25, 1), done.log_density);
}
