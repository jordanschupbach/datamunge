#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/hungarian.hpp>
#include <limits>
#include <numeric>
#include <set>
using namespace datamunge::algorithms;
namespace {
double brute(const std::vector<std::vector<double>>& c) {
  std::vector<std::size_t> p(c.size());
  std::iota(p.begin(), p.end(), 0);
  double best = std::numeric_limits<double>::infinity();
  do {
    double s = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
      s += c[i][p[i]];
    best = std::min(best, s);
  } while (std::next_permutation(p.begin(), p.end()));
  return best;
}
void certificate(const std::vector<std::vector<double>>& c, const AssignmentTrace& t) {
  const auto  n = c.size();
  const auto& r = t.run;
  ASSERT_EQ(r.result.assignment.size(), n);
  std::set<std::size_t> columns;
  double                primal = 0, dual = 0;
  for (std::size_t i = 0; i < n; ++i) {
    columns.insert(r.result.assignment[i]);
    primal += c[i][r.result.assignment[i]];
    dual += r.row_potentials[i] + r.column_potentials[i];
  }
  EXPECT_EQ(columns.size(), n);
  EXPECT_DOUBLE_EQ(primal, brute(c));
  EXPECT_DOUBLE_EQ(primal, r.result.cost);
  EXPECT_NEAR(primal, dual, 1e-10);
  std::size_t reprices = 0, augments = 0;
  for (const auto& e : t.events) {
    ASSERT_EQ(e.row_potentials.size(), n);
    ASSERT_EQ(e.column_to_row.size(), n);
    ASSERT_LT(e.introduced_row, n);
    ASSERT_LT(e.next_column, n);
    for (std::size_t i = 0; i <= e.introduced_row; ++i)
      for (std::size_t j = 0; j < n; ++j)
        EXPECT_LE(e.row_potentials[i] + e.column_potentials[j], c[i][j] + 1e-10);
    std::set<std::size_t> rows;
    for (std::size_t j = 0; j < n; ++j)
      if (e.column_to_row[j] != kAssignmentUnmatched) {
        auto i = e.column_to_row[j];
        ASSERT_LE(i, e.introduced_row);
        rows.insert(i);
        EXPECT_NEAR(e.row_potentials[i] + e.column_potentials[j], c[i][j], 1e-10);
      }
    if (e.kind == AssignmentEventKind::Reprice) {
      ++reprices;
      EXPECT_EQ(rows.size(), e.introduced_row);
      EXPECT_EQ(e.reached_columns[e.next_column], 0);
    } else {
      ++augments;
      EXPECT_EQ(rows.size(), e.introduced_row + 1);
      EXPECT_EQ(e.delta, 0);
    }
  }
  EXPECT_EQ(augments, n);
  EXPECT_EQ(reprices, r.work.reprices);
  EXPECT_EQ(t.events.size(), reprices + n);
  EXPECT_EQ(hungarian(c).assignment, r.result.assignment);
  EXPECT_EQ(hungarian_run(c).result.assignment, r.result.assignment);
}
} // namespace
TEST(HungarianTrace, ExhaustiveSignedThreeByThreeAndCertificates) {
  for (unsigned code = 0; code < 19683; ++code) {
    auto                             x = code;
    std::vector<std::vector<double>> c(3, std::vector<double>(3));
    for (auto& row : c)
      for (auto& a : row) {
        a = int(x % 3) - 1;
        x /= 3;
      }
    certificate(c, hungarian_trace(c));
  }
}
TEST(HungarianTrace, ActualRepriceAndPathChanges) {
  const auto t = hungarian_trace({{9, 2, 7}, {6, 4, 3}, {5, 8, 1}});
  certificate(t.costs, t);
  EXPECT_EQ(t.run.result.assignment, (std::vector<std::size_t>{1, 0, 2}));
  EXPECT_EQ(t.run.result.cost, 9);
  const auto swap = hungarian_trace({{1, 2}, {1, 100}});
  certificate(swap.costs, swap);
  EXPECT_EQ(swap.run.result.assignment, (std::vector<std::size_t>{1, 0}));
  EXPECT_GT(swap.run.work.path_writes, 2);
  EXPECT_EQ(hungarian({{0, 0}, {0, 0}}).assignment, (std::vector<std::size_t>{0, 1}));
}
TEST(HungarianTrace, CompleteCapsAndOwnedStates) {
  auto c     = std::vector<std::vector<double>>{{1, 2}, {1, 100}};
  auto t     = hungarian_trace(c);
  auto units = t.events.size() * 9;
  EXPECT_NO_THROW(hungarian_trace(c, {4, units}));
  EXPECT_THROW(hungarian_trace(c, {4, units - 1}), std::length_error);
  EXPECT_THROW(hungarian_trace(c, {3, 0}), std::length_error);
  EXPECT_NO_THROW(hungarian_run(c, {4, 1}));
  EXPECT_NO_THROW(hungarian_trace(c, {0, 0}));
  c[0][0] = 100;
  EXPECT_EQ(t.costs[0][0], 1);
  auto last                          = t.events.back().row_potentials;
  t.events.front().row_potentials[0] = 999;
  EXPECT_EQ(t.events.back().row_potentials, last);
}
TEST(HungarianTrace, ShapeNonfiniteAndProducedOverflow) {
  const auto inf = std::numeric_limits<double>::infinity(),
             nan = std::numeric_limits<double>::quiet_NaN(),
             max = std::numeric_limits<double>::max();
  EXPECT_THROW(hungarian({}), std::invalid_argument);
  EXPECT_THROW(hungarian({{1, 2}}), std::invalid_argument);
  EXPECT_THROW(hungarian({{1, 2}, {3}}), std::invalid_argument);
  EXPECT_THROW(hungarian({{inf}}), std::invalid_argument);
  EXPECT_THROW(hungarian_trace({{nan}}), std::invalid_argument);
  EXPECT_THROW(hungarian_run({{-inf}}), std::invalid_argument);
  EXPECT_THROW(hungarian({{max, max}, {max, max}}), std::overflow_error);
  EXPECT_THROW(hungarian({{-max, max}, {max, -max}}), std::overflow_error);
  EXPECT_EQ(hungarian({{-3}}).cost, -3);
}

#include <datamunge/algorithms/hungarian_analysis.hpp>
TEST(HungarianTrace, NativeDiagnosticsDiagramsAndWorkFamilies) {
  const auto c = assignment_example_costs();
  auto       r = hungarian_run(c);
  const auto d = assignment_diagnostics(c, r);
  EXPECT_EQ(d.primal, 9);
  EXPECT_EQ(d.dual, 9);
  EXPECT_EQ(d.gap, 0);
  EXPECT_EQ(d.minimum_slack, 0);
  EXPECT_EQ(d.maximum_matched_residual, 0);
  EXPECT_FALSE(assignment_diagram(c, r).to_svg().empty());
  EXPECT_THROW(assignment_diagram(c, r, 2), std::length_error);
  auto wrong        = r;
  wrong.result.cost = 100;
  EXPECT_EQ(assignment_diagnostics(c, wrong).primal, 9);
  wrong.result.assignment = {0, 0, 2};
  EXPECT_THROW(assignment_diagnostics(c, wrong), std::invalid_argument);
  wrong = r;
  wrong.row_potentials.pop_back();
  EXPECT_THROW(assignment_diagnostics(c, wrong), std::invalid_argument);
  wrong                      = r;
  wrong.column_potentials[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(assignment_diagnostics(c, wrong), std::invalid_argument);
  auto bad = c;
  bad[2].pop_back();
  EXPECT_THROW(assignment_diagnostics(bad, r), std::invalid_argument);
  bad       = c;
  bad[2][2] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(assignment_diagnostics(bad, r), std::invalid_argument);
  for (std::size_t n = 1; n <= 40; ++n) {
    auto a = hungarian_run(assignment_work_costs(n));
    auto b = hungarian_run(assignment_work_costs(n, true));
    EXPECT_EQ(a.work.slack_evaluations, n * n);
    EXPECT_EQ(a.work.reprices, n);
    EXPECT_EQ(b.work.slack_evaluations, n * (n + 1) * (2 * n + 1) / 6);
    EXPECT_EQ(b.work.reprices, n * (n + 1) / 2);
  }
  EXPECT_THROW(assignment_work_costs(0), std::invalid_argument);
  EXPECT_THROW(assignment_work_costs(3, false, 8), std::length_error);
  EXPECT_THROW(assignment_work_costs(std::numeric_limits<std::size_t>::max()), std::length_error);
}
