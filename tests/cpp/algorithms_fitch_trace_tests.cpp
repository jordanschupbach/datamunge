#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <datamunge/algorithms/maximum_parsimony.hpp>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(FitchTrace, AllThreeStateFourTaxonColumnsAndThreeTopologiesAgainstEveryInternalAssignment) {
  for (auto ids : std::vector<std::array<int, 4>>{{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}}) {
    auto tree = parsimony_four_taxon_tree(ids[0], ids[1], ids[2], ids[3]);
    for (int encoded = 0; encoded < 81; ++encoded) {
      auto               bits = encoded;
      std::array<int, 4> leaves;
      for (auto& x : leaves) {
        x = bits % 3;
        bits /= 3;
      }
      auto          states = parsimony_four_taxon_column(leaves);
      auto          t      = fitch_trace(tree, states);
      int           best   = INT_MAX;
      std::uint32_t roots  = 0;
      for (int internal = 0; internal < 27; ++internal) {
        auto b          = internal;
        auto assignment = states;
        for (int i = 4; i < 7; ++i) {
          assignment[i] = b % 3;
          b /= 3;
        }
        int score = 0;
        for (int node = 4; node < 7; ++node) {
          auto [l, r] = tree.children[node];
          score += assignment[node] != assignment[l];
          score += assignment[node] != assignment[r];
        }
        if (score < best) {
          best  = score;
          roots = 0;
        }
        if (score == best)
          roots |= std::uint32_t{1} << assignment[6];
      }
      EXPECT_EQ(t.analysis.score, best);
      EXPECT_EQ(t.analysis.masks[6], roots);
      EXPECT_EQ(fitch_small_parsimony(tree, states), best);
      EXPECT_EQ(t.captured_entries, 49);
      EXPECT_EQ(t.steps.size(), 7);
      EXPECT_EQ(t.edges.size(), 6);
      int                        running = 0;
      std::vector<std::uint32_t> masks(7);
      for (auto& s : t.steps) {
        EXPECT_EQ(s.before, running);
        auto [l, r] = tree.children[s.node];
        EXPECT_EQ(s.left, l);
        EXPECT_EQ(s.right, r);
        if (l == -1) {
          EXPECT_EQ(s.mask, std::uint32_t{1} << states[s.node]);
          EXPECT_EQ(s.left_mask, 0);
          EXPECT_EQ(s.right_mask, 0);
        } else {
          EXPECT_EQ(s.left_mask, masks[l]);
          EXPECT_EQ(s.right_mask, masks[r]);
          auto inter = masks[l] & masks[r];
          EXPECT_EQ(s.intersection, inter);
          if (!inter)
            ++running;
          EXPECT_EQ(s.mask, inter ? inter : masks[l] | masks[r]);
        }
        masks[s.node] = s.mask;
        EXPECT_EQ(s.after, running);
      }
      EXPECT_EQ(masks, t.analysis.masks);
      EXPECT_EQ(running, best);
      std::size_t changes = 0;
      for (auto& e : t.edges) {
        EXPECT_EQ(e.parent_state, t.analysis.states[e.parent]);
        EXPECT_EQ(e.child_state, t.analysis.states[e.child]);
        EXPECT_EQ(e.changed, e.parent_state != e.child_state);
        changes += e.changed;
        EXPECT_TRUE(t.analysis.masks[e.child] & (std::uint32_t{1} << e.child_state));
      }
      EXPECT_EQ(changes, best);
      EXPECT_EQ(t.analysis.work.nodes, 7);
      EXPECT_EQ(t.analysis.work.leaves, 4);
      EXPECT_EQ(t.analysis.work.intersections + t.analysis.work.unions, 3);
      EXPECT_EQ(t.analysis.work.unions, best);
      EXPECT_EQ(t.analysis.work.changes, best);
    }
  }
}
TEST(FitchTrace, OriginalSitesHighBitEmptyQuotasOwnershipAndCompleteGuards) {
  auto tree    = parsimony_four_taxon_tree(0, 1, 2, 3);
  auto columns = parsimony_demo_columns();
  EXPECT_EQ(parsimony_site_scores(tree, columns), (std::vector<int>{1, 1, 1, 1, 1}));
  EXPECT_EQ(parsimony_score(tree, columns), 5);
  EXPECT_EQ(parsimony_score(parsimony_four_taxon_tree(0, 2, 1, 3), columns), 8);
  EXPECT_EQ(parsimony_score(parsimony_four_taxon_tree(0, 3, 1, 2), columns), 8);
  auto high = fitch_trace({0, {{-1, -1}}}, {31});
  EXPECT_EQ(high.analysis.masks[0], 0x80000000U);
  EXPECT_EQ(high.analysis.states[0], 31);
  EXPECT_EQ(high.analysis.score, 0);
  EXPECT_NO_THROW(fitch_trace({0, {{-1, -1}}}, {0}, 7));
  EXPECT_THROW(fitch_trace({0, {{-1, -1}}}, {0}, 6), std::length_error);
  EXPECT_NO_THROW(fitch_trace({0, {{-1, -1}}}, {0}, 0));
  EXPECT_EQ(fitch_trace({-1, {}}, {}, 1).captured_entries, 1);
  EXPECT_TRUE(fitch_analyze({-1, {}}, {}).masks.empty());
  auto owned       = fitch_trace(tree, columns[0]);
  tree.children[4] = {0, 3};
  columns[0][0]    = 31;
  EXPECT_EQ(owned.tree.children[4], (std::pair<int, int>{0, 1}));
  EXPECT_EQ(owned.leaf_states[0], 0);
  for (auto bad : std::vector<ParsimonyTree>{
           {0, {}},
           {-1, {{-1, -1}}},
           {0, {{-1, 0}}},
           {0, {{1, 1}, {-1, -1}}},
           {0, {{1, 2}, {-1, -1}, {0, 1}}},
           {0, {{-1, -1}, {2, 3}, {1, 4}, {-1, -1}, {-1, -1}}}}) {
    EXPECT_THROW(parsimony_score(bad, {}), std::invalid_argument);
    EXPECT_THROW(fitch_trace(bad, {}, 1), std::invalid_argument);
  }
  EXPECT_THROW(fitch_analyze({0, {{-1, -1}}}, {}), std::invalid_argument);
  EXPECT_THROW(fitch_analyze({0, {{-1, -1}}}, {32}), std::invalid_argument);
  EXPECT_THROW(fitch_analyze({0, {{-1, -1}}}, {-1}), std::invalid_argument);
  EXPECT_THROW(parsimony_site_scores({0, {{-1, -1}}}, {{0}, {32}}), std::invalid_argument);
  EXPECT_THROW(parsimony_four_taxon_tree(0, 0, 2, 3), std::invalid_argument);
  EXPECT_THROW(parsimony_four_taxon_column({0, 1, 2, 32}), std::invalid_argument);
}
TEST(FitchTrace, IterativeVeryDeepFullBinaryTree) {
  const int     internals = 50000;
  ParsimonyTree tree;
  tree.root = 2 * internals;
  tree.children.assign(2 * internals + 1, {-1, -1});
  std::vector<int> states(2 * internals + 1, -1);
  for (int i = 0; i <= internals; ++i)
    states[i] = 0;
  for (int k = 0; k < internals; ++k)
    tree.children[internals + 1 + k] = {k ? internals + k : 0, k + 1};
  auto a = fitch_analyze(tree, states);
  EXPECT_EQ(a.score, 0);
  EXPECT_EQ(a.work.nodes, 100001);
  EXPECT_EQ(a.work.edges, 100000);
  EXPECT_TRUE(std::all_of(a.masks.begin(), a.masks.end(), [](auto x) { return x == 1; }));
}
