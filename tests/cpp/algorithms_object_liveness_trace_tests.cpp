#include <gtest/gtest.h>

#include <datamunge/algorithms/object_liveness.hpp>
using namespace datamunge::algorithms;
TEST(ObjectLivenessTrace, AllThreeVertexDirectedGraphsAndRootSets) {
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
      auto t = object_liveness_trace(g);
      EXPECT_EQ(t.analysis.marked, expected);
      EXPECT_EQ(reachable_from_roots(g), expected);
      std::vector<char> marked(3, 0);
      std::vector<int>  stack;
      std::size_t       pops = 0, refs = 0, pushes = 0, maxstack = 0, root_slots = 0;
      int               current = -1;
      std::size_t       field   = 0;
      for (auto& e : t.events) {
        EXPECT_EQ(e.stack_before, stack.size());
        if (e.kind == ObjectMarkEventKind::Pop) {
          if (current >= 0)
            EXPECT_EQ(field, g.refs[current].size());
          ASSERT_FALSE(stack.empty());
          EXPECT_EQ(e.object, stack.back());
          stack.pop_back();
          current = e.object;
          field   = 0;
          ++pops;
          EXPECT_TRUE(e.already_marked);
          EXPECT_FALSE(e.pushed);
        } else {
          if (e.kind == ObjectMarkEventKind::Root) {
            EXPECT_EQ(e.slot, root_slots);
            EXPECT_EQ(e.object, g.roots[root_slots++]);
          } else {
            ASSERT_GE(current, 0);
            EXPECT_EQ(e.parent, current);
            EXPECT_EQ(e.slot, field);
            EXPECT_EQ(e.object, g.refs[current][field++]);
            ++refs;
          }
          EXPECT_EQ(e.already_marked, marked[e.object] != 0);
          EXPECT_EQ(e.pushed, marked[e.object] == 0);
          if (e.pushed) {
            marked[e.object] = 1;
            stack.push_back(e.object);
            ++pushes;
            maxstack = std::max(maxstack, stack.size());
          }
        }
        EXPECT_EQ(e.stack_after, stack.size());
      }
      EXPECT_TRUE(stack.empty());
      EXPECT_EQ(marked, expected);
      EXPECT_EQ(t.analysis.work.pops, pops);
      EXPECT_EQ(t.analysis.work.references, refs);
      EXPECT_EQ(t.analysis.work.pushes, pushes);
      EXPECT_EQ(t.analysis.work.marks, pushes);
      EXPECT_EQ(t.analysis.work.maximum_stack, maxstack);
      EXPECT_EQ(t.analysis.work.roots, g.roots.size());
      std::size_t edges = 0;
      for (auto& r : g.refs)
        edges += r.size();
      EXPECT_EQ(t.analysis.work.validation.references, edges);
      EXPECT_EQ(t.captured_entries, 6 + edges + g.roots.size() + t.events.size());
    }
}
TEST(ObjectLivenessTrace, CyclesAliasesDeepChainCompleteGuardsQuotaAndOwnership) {
  ObjectGraph g{4, {{1, 1, 0}, {2, 2}, {1}, {}}, {0, 0, 2}, {-5, 0}};
  auto        t = object_liveness_trace(g);
  EXPECT_EQ(t.analysis.marked, (std::vector<char>{1, 1, 1, 0}));
  EXPECT_EQ(t.analysis.work.pushes, 3);
  EXPECT_EQ(t.analysis.work.references, 6);
  EXPECT_EQ(t.analysis.work.roots, 3);
  EXPECT_EQ(t.captured_entries, 31);
  EXPECT_THROW(object_liveness_trace(g, 30), std::length_error);
  EXPECT_EQ(object_liveness_trace(g, 31).captured_entries, 31);
  g.refs[0].clear();
  EXPECT_EQ(t.graph.refs[0].size(), 3);
  EXPECT_EQ(t.graph.sizes, (std::vector<int>{-5, 0}));
  EXPECT_THROW(object_liveness_trace({-1, {}, {}, {}}, 1), std::invalid_argument);
  EXPECT_THROW(object_liveness_trace({2, {{}, {}}, {-1}, {}}, 1), std::invalid_argument);
  EXPECT_THROW(object_liveness_trace({2, {{}, {2}}, {0}, {}}, 1), std::invalid_argument);
  EXPECT_THROW(object_liveness_trace({2, {{}}, {0}, {}}, 1), std::invalid_argument);
  EXPECT_TRUE(object_liveness_trace({}).events.empty());
  ObjectGraph chain;
  chain.num_objects = 100000;
  chain.refs.resize(chain.num_objects);
  chain.roots = {0};
  for (int i = 0; i + 1 < chain.num_objects; ++i)
    chain.refs[i] = {i + 1};
  auto a = object_liveness_analyze(chain);
  EXPECT_EQ(a.work.marks, 100000);
  EXPECT_EQ(a.work.maximum_stack, 1);
}
