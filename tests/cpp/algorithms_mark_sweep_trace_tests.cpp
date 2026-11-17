#include <gtest/gtest.h>

#include <datamunge/algorithms/mark_sweep.hpp>
using namespace datamunge::algorithms;
TEST(MarkSweepTrace, AllThreeVertexDirectedGraphsAndRootSets) {
  for (unsigned mask = 0; mask < 512; ++mask)
    for (unsigned roots = 0; roots < 8; ++roots) {
      ObjectGraph g{3, std::vector<std::vector<int>>(3), {}, {}};
      for (int i = 0; i < 3; ++i) {
        if (roots & (1u << i))
          g.roots.push_back(i);
        for (int j = 0; j < 3; ++j)
          if (mask & (1u << (3 * i + j)))
            g.refs[i].push_back(j);
      }
      std::vector<char> expected(3, 0);
      for (auto r : g.roots)
        expected[r] = 1;
      for (int pass = 0; pass < 3; ++pass)
        for (int i = 0; i < 3; ++i)
          if (expected[i])
            for (int j : g.refs[i])
              expected[j] = 1;
      auto t = mark_sweep_trace(g);
      EXPECT_EQ(t.analysis.result.marked, expected);
      EXPECT_EQ(t.marking.analysis.marked, expected);
      std::vector<int> live, freed;
      for (int i = 0; i < 3; ++i) {
        auto& s = t.sweep[i];
        EXPECT_EQ(s.object, i);
        EXPECT_EQ(s.live, expected[i] != 0);
        EXPECT_EQ(s.live_before, live.size());
        EXPECT_EQ(s.freed_before, freed.size());
        (expected[i] ? live : freed).push_back(i);
        EXPECT_EQ(s.live_after, live.size());
        EXPECT_EQ(s.freed_after, freed.size());
      }
      EXPECT_EQ(t.analysis.result.live, live);
      EXPECT_EQ(t.analysis.result.freed, freed);
      EXPECT_EQ(t.analysis.work.scanned, 3);
      EXPECT_EQ(t.analysis.work.live, live.size());
      EXPECT_EQ(t.analysis.work.reclaimed, freed.size());
      EXPECT_EQ(t.captured_entries, t.marking.captured_entries + 3);
      EXPECT_EQ(mark_sweep(g).marked, expected);
    }
}
TEST(MarkSweepTrace, OriginalCyclicGarbageCompleteGuardsAndOwnership) {
  auto g = mark_sweep_demo_graph();
  auto t = mark_sweep_trace(g);
  EXPECT_EQ(t.analysis.result.live, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(t.analysis.result.freed, (std::vector<int>{3, 4, 5, 6, 7}));
  EXPECT_EQ(t.analysis.work.marking.references, 2);
  EXPECT_EQ(t.analysis.work.marking.validation.references, 5);
  EXPECT_THROW(mark_sweep_trace(g, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(mark_sweep_trace(g, t.captured_entries).captured_entries, t.captured_entries);
  g.sizes = {-10, 0};
  EXPECT_EQ(mark_sweep(g).marked, t.analysis.result.marked);
  g.roots = {3};
  EXPECT_EQ(t.marking.graph.roots, (std::vector<int>{0}));
  EXPECT_EQ(mark_sweep(g).live, (std::vector<int>{3, 4}));
  EXPECT_TRUE(mark_sweep_trace({}).analysis.result.live.empty());
  EXPECT_THROW(mark_sweep_trace({2, {{}, {2}}, {0}, {}}, 1), std::invalid_argument);
  EXPECT_EQ(mark_sweep_trace(mark_sweep_demo_graph(), 0).analysis.result.freed.size(), 5);
}
