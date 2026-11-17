#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/nonblocking_switch.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(SwitchRoutesTrace, ExhaustiveSmallMultigraphsAndIndependentColorCertificates) {
  for (int n = 0; n <= 4; ++n) {
    int count = 1 << (2 * n);
    for (int code = 0; code < count; ++code) {
      int                                              bits = code;
      std::vector<std::pair<std::size_t, std::size_t>> c;
      int                                              in[2]{}, out[2]{};
      for (int i = 0; i < n; ++i) {
        int edge = bits % 4;
        bits /= 4;
        c.emplace_back(edge / 2, edge % 2);
        ++in[edge / 2];
        ++out[edge % 2];
      }
      bool valid = std::max(in[0], in[1]) <= 2 && std::max(out[0], out[1]) <= 2;
      if (!valid) {
        EXPECT_THROW(nonblocking_switch_routes(2, 2, c), std::invalid_argument);
        continue;
      }
      auto  t      = nonblocking_switch_trace(2, 2, c);
      auto& r      = t.analysis.routes;
      bool  oracle = false;
      for (int colors = 0; colors < (1 << n); ++colors) {
        bool okay = true;
        for (int i = 0; i < n; ++i)
          for (int j = i + 1; j < n; ++j)
            if (((colors >> i) & 1) == ((colors >> j) & 1) &&
                (c[i].first == c[j].first || c[i].second == c[j].second))
              okay = false;
        if (okay) {
          oracle = true;
          std::vector<std::size_t> choices;
          for (int i = 0; i < n; ++i)
            choices.push_back((colors >> i) & 1);
          EXPECT_TRUE(nonblocking_switch_feasible(2, 2, c, choices));
        }
      }
      EXPECT_TRUE(oracle);
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
          if (c[i].first == c[j].first || c[i].second == c[j].second)
            EXPECT_NE(r[i], r[j]);
      EXPECT_EQ(t.analysis.assignments, n);
      EXPECT_EQ(t.analysis.queue_pops, n);
      EXPECT_EQ(t.analysis.incidence_reads, t.scans.size());
      EXPECT_EQ(t.captured_entries, 2 + 4 * n + t.scans.size());
      EXPECT_EQ(t.assignments.size(), n);
      for (auto& a : t.assignments) {
        EXPECT_EQ(a.middle, r[a.call]);
        if (a.component_start) {
          EXPECT_EQ(a.middle, 0);
          EXPECT_EQ(a.parent, n);
        } else {
          EXPECT_NE(a.middle, r[a.parent]);
          EXPECT_EQ(
              a.via_input ? c[a.call].first : c[a.call].second,
              a.via_input ? c[a.parent].first : c[a.parent].second);
        }
      }
      for (auto& q : t.scans) {
        EXPECT_EQ(
            q.via_input ? c[q.from].first : c[q.from].second,
            q.via_input ? c[q.candidate].first : c[q.candidate].second);
        EXPECT_EQ(q.assigned, q.color_before == 2);
      }
      EXPECT_NO_THROW(nonblocking_switch_trace(2, 2, c, {0, 0, t.captured_entries}));
      EXPECT_THROW(
          nonblocking_switch_trace(2, 2, c, {0, 0, t.captured_entries - 1}), std::length_error);
    }
  }
}
TEST(SwitchRoutesTrace, OriginalCyclesParallelCallsCompletePoliciesAndOwnership) {
  auto c = nonblocking_switch_demo_calls();
  auto t = nonblocking_switch_trace(2, 2, c);
  EXPECT_EQ(t.analysis.routes, (std::vector<std::size_t>{0, 1, 0, 1}));
  EXPECT_EQ(t.analysis.components, 1);
  EXPECT_EQ(t.analysis.incidence_reads, 16);
  EXPECT_EQ(t.captured_entries, 34);
  c[0] = {1, 1};
  EXPECT_EQ(t.connections[0], (std::pair<std::size_t, std::size_t>{0, 0}));
  for (int n = 1; n <= 64; ++n) {
    auto calls = nonblocking_switch_cycle_calls(n);
    auto r     = nonblocking_switch_analyze(n, n, calls);
    EXPECT_EQ(r.components, 1);
    EXPECT_EQ(r.incidence_reads, 8 * n);
    EXPECT_TRUE(nonblocking_switch_feasible(n, n, calls, r.routes));
  }
  EXPECT_TRUE(nonblocking_switch_routes(0, 0, {}).empty());
  EXPECT_THROW(nonblocking_switch_trace(1, 1, {{1, 0}}, {1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(
      nonblocking_switch_trace(2, 2, nonblocking_switch_demo_calls(), {3, 0, 0}),
      std::length_error);
  EXPECT_THROW(
      nonblocking_switch_trace(2, 2, nonblocking_switch_demo_calls(), {0, 3, 0}),
      std::length_error);
  EXPECT_THROW(
      nonblocking_switch_routes(std::numeric_limits<std::size_t>::max(), 1, {}), std::length_error);
  EXPECT_TRUE(nonblocking_switch_feasible(2, 2, nonblocking_switch_demo_calls(), {1, 0, 1, 0}));
  EXPECT_FALSE(nonblocking_switch_feasible(2, 2, nonblocking_switch_demo_calls(), {0, 0, 0, 0}));
  EXPECT_FALSE(nonblocking_switch_feasible(2, 2, nonblocking_switch_demo_calls(), {0, 1}));
  EXPECT_FALSE(nonblocking_switch_feasible(2, 2, nonblocking_switch_demo_calls(), {2, 1, 0, 1}));
}
