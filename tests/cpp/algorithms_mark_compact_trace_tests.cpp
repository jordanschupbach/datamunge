#include <gtest/gtest.h>

#include <datamunge/algorithms/mark_compact.hpp>
using namespace datamunge::algorithms;
TEST(MarkCompactTrace, AllThreeVertexDirectedGraphsRootsAndActualRewrites) {
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
      auto  t = mark_compact_trace(g);
      auto& a = t.analysis;
      EXPECT_EQ(a.result.marked, expected);
      EXPECT_EQ(t.marking.analysis.marked, expected);
      std::vector<int> map(3, -1), order;
      for (int i = 0; i < 3; ++i)
        if (expected[i]) {
          map[i] = int(order.size());
          order.push_back(i);
        }
      EXPECT_EQ(a.result.new_address, map);
      EXPECT_EQ(a.result.order, order);
      std::size_t references = 0;
      for (std::size_t i = 0; i < order.size(); ++i) {
        std::vector<int> fields;
        for (int child : g.refs[order[i]])
          fields.push_back(map[child]);
        EXPECT_EQ(a.result.updated_refs[i], fields);
        references += fields.size();
      }
      for (std::size_t i = 0; i < g.roots.size(); ++i)
        EXPECT_EQ(a.relocated_roots[i], map[g.roots[i]]);
      EXPECT_EQ(t.forwarding.size(), 3);
      for (auto& f : t.forwarding) {
        EXPECT_EQ(f.assigned, map[f.object]);
        EXPECT_EQ(f.live, expected[f.object] != 0);
        EXPECT_EQ(f.next_after, f.next_before + f.live);
      }
      for (auto& r : t.references) {
        EXPECT_EQ(r.owner_new, map[r.owner_old]);
        EXPECT_EQ(r.target_old, g.refs[r.owner_old][r.field]);
        EXPECT_EQ(r.target_new, map[r.target_old]);
        EXPECT_EQ(a.result.updated_refs[r.owner_new][r.field], r.target_new);
      }
      for (auto& r : t.roots) {
        EXPECT_EQ(r.old_id, g.roots[r.slot]);
        EXPECT_EQ(r.new_slot, map[r.old_id]);
      }
      EXPECT_EQ(a.work.scanned, 3);
      EXPECT_EQ(a.work.forwarded, order.size());
      EXPECT_EQ(a.work.references, references);
      EXPECT_EQ(a.work.rewritten, references);
      EXPECT_EQ(a.work.roots, g.roots.size());
      EXPECT_EQ(t.captured_entries, t.marking.captured_entries + 3 + references + g.roots.size());
    }
}
TEST(MarkCompactTrace, OriginalAliasesCyclesIgnoredSizesQuotaAndOwnership) {
  auto g = mark_compact_demo_graph();
  auto t = mark_compact_trace(g);
  EXPECT_EQ(t.analysis.result.order, (std::vector<int>{0, 2, 4, 6, 8}));
  EXPECT_EQ(t.analysis.result.new_address, (std::vector<int>{0, -1, 1, -1, 2, -1, 3, -1, 4, -1}));
  EXPECT_EQ(
      t.analysis.result.updated_refs, (std::vector<std::vector<int>>{{1}, {2}, {3}, {4}, {}}));
  EXPECT_EQ(t.analysis.relocated_roots, (std::vector<int>{0}));
  EXPECT_THROW(mark_compact_trace(g, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(mark_compact_trace(g, t.captured_entries).captured_entries, t.captured_entries);
  g.sizes = {-100, 0};
  EXPECT_EQ(mark_compact(g).new_address, t.analysis.result.new_address);
  g.refs[0].clear();
  EXPECT_EQ(t.marking.graph.refs[0], (std::vector<int>{2}));
  ObjectGraph aliases{6, {{2, 2}, {}, {4}, {5}, {2}, {}}, {0, 0, 4}, {}};
  auto        a = mark_compact_analyze(aliases);
  EXPECT_EQ(a.result.order, (std::vector<int>{0, 2, 4}));
  EXPECT_EQ(a.result.updated_refs, (std::vector<std::vector<int>>{{1, 1}, {2}, {1}}));
  EXPECT_EQ(a.relocated_roots, (std::vector<int>{0, 0, 2}));
  EXPECT_TRUE(mark_compact_trace({}).analysis.result.order.empty());
  EXPECT_THROW(mark_compact_trace({2, {{}, {2}}, {0}, {}}, 1), std::invalid_argument);
  EXPECT_EQ(mark_compact_trace(mark_compact_demo_graph(), 0).analysis.result.order.size(), 5);
}
