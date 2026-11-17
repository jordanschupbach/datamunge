#pragma once

/// @file maximum_parsimony.hpp
/// @brief Canonical maximum parsimony API, producing histories and checked native contracts.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned rooted full binary tree with integer node ids.
/// @details Leaves have children(-1,-1). Every other node has two distinct valid children. Every node belongs to exactly this rooted tree, with no cycles or shared descendants. Empty tree is root-1 with no nodes. The legacy value-initialized default root remains0, so construct the empty form explicitly.
/// @par Example
/// @code
/// auto t=dm::algorithms::parsimony_four_taxon_tree(0,1,2,3);
/// std::print("{}\n",t.root); // 6
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
struct ParsimonyTree {
  int root{0}; ///< Root node id, or -1 for an empty tree.
  std::vector<std::pair<int, int>>
      children; ///< Ordered left/right children, or exact leaf marker(-1,-1).
};

/// @brief Actual checked postorder and witness-assignment counts.
/// @details Validation scans all nodes before evaluation; selected counters describe the subsequent canonical scoring/witness passes.
/// @par Example
/// @code
/// auto t=dm::algorithms::parsimony_four_taxon_tree(0,1,2,3);
/// auto a=dm::algorithms::fitch_analyze(t,dm::algorithms::parsimony_four_taxon_column({0,0,1,1}));
/// std::print("{}\n",a.work.nodes); // 7
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
struct FitchWork {
  std::size_t nodes{0};         ///< Actual postorder node evaluations.
  std::size_t leaves{0};        ///< Actual singleton-state leaf reads.
  std::size_t intersections{0}; ///< Actual nonempty child-set intersection decisions.
  std::size_t unions{0};        ///< Actual empty-intersection unions, each adding one score.
  std::size_t edges{0};         ///< Actual top-down child assignments.
  std::size_t changes{0};       ///< Actual unequal parent/child witness states.
};

/// @brief Same-run minimum score, actual node masks and one minimum-change witness.
/// @details Unit unordered changes are counted per tree edge; this is small parsimony on a fixed full binary topology. It does not search all trees, fit branch lengths or model substitution probabilities.
/// @par Example
/// @code
/// auto t=dm::algorithms::parsimony_four_taxon_tree(0,1,2,3);
/// auto a=dm::algorithms::fitch_analyze(t,dm::algorithms::parsimony_four_taxon_column({0,0,1,1}));
/// std::print("{}\n",a.score); // 1
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
struct FitchAnalysis {
  int                        score{0}; ///< Exact minimum unordered unit-change edge count.
  std::vector<std::uint32_t> masks;    ///< Actual postorder feasible state masks, indexed by node.
  std::vector<int>           states;   ///< Actual chosen minimum-change states, indexed by node.
  FitchWork                  work;     ///< Actual canonical scoring and witness operations.
};

/// @brief One actual postorder state-set write and score update.
/// @details Leaves have zero child masks and intersection; their actual singleton mask is still recorded. Internal writes use actual committed child masks.
/// @par Example
/// @code
/// auto t=dm::algorithms::fitch_trace({0,{{-1,-1}}},{31});
/// std::print("{}\n",t.steps[0].mask); // 2147483648
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
struct FitchStep {
  int           node{-1};        ///< Actual evaluated node.
  int           left{-1};        ///< Actual left child, -1 at a leaf.
  int           right{-1};       ///< Actual right child, -1 at a leaf.
  std::uint32_t left_mask{0};    ///< Actual committed left-child mask, zero at a leaf.
  std::uint32_t right_mask{0};   ///< Actual committed right-child mask, zero at a leaf.
  std::uint32_t intersection{0}; ///< Actual bitwise child intersection, zero at a leaf.
  std::uint32_t mask{0};         ///< Actual mask written.
  int           before{0};       ///< Actual accumulated score before this node.
  int           after{0};        ///< Actual score after this node.
};

/// @brief One actual top-down witness child-state assignment.
/// @details Root chooses the smallest set bit. A child retains its parent state if present in its mask, otherwise chooses its smallest set bit; edge change is exact inequality.
/// @par Example
/// @code
/// auto tree=dm::algorithms::parsimony_four_taxon_tree(0,1,2,3);
/// auto t=dm::algorithms::fitch_trace(tree,dm::algorithms::parsimony_four_taxon_column({0,0,1,1}));
/// std::print("{}\n",t.edges.size()); // 6
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
struct FitchEdge {
  int  parent{-1};      ///< Actual assigned parent.
  int  child{-1};       ///< Actual assigned child.
  int  parent_state{0}; ///< Actual parent witness state.
  int  child_state{0};  ///< Actual chosen child witness state.
  bool changed{false};  ///< Actual inequality of the two chosen states.
};

/// @brief Owned inputs and actual complete scoring/witness history.
/// @details Logical quota counts one root scalar, two child-id scalars per node, each input state, final mask, chosen state, node step and edge record. For N>0 this is7N; empty tree costs one. Fields, vector headers, capacities, runtime traversal storage and bytes are excluded.
/// @par Example
/// @code
/// auto t=dm::algorithms::fitch_trace({0,{{-1,-1}}},{0});
/// std::print("{}\n",t.captured_entries); // 7
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
struct FitchTrace {
  ParsimonyTree tree; ///< Owned original complete tree.
  std::vector<int>
      leaf_states; ///< Owned original node-indexed input state vector, including ignored internal values.
  FitchAnalysis          analysis; ///< Actual same-run masks, witness, score and work.
  std::vector<FitchStep> steps;    ///< Actual left-before-right postorder writes.
  std::vector<FitchEdge>
      edges; ///< Actual child assignments; parent nodes use left-before-right DFS, each assigning left then right.
  std::size_t captured_entries{0}; ///< Complete logical entry count.
};

/// @brief Score a fixed full binary topology by exact unordered unit-change Fitch sets.
/// @details Complete tree validation precedes state access: root/children, unique parents, no shared child or cycle, and complete root reachability. Leaf states are0..31, using safe uint32 singleton shifts; internal input states are ignored but the vector must have N entries. Iterative left-before-right postorder computes real masks and union increments, then a top-down pass constructs one actual minimum-change witness. Time and storage O(N), including validation and returned masks/witness. Empty tree uses root-1 and an empty state vector.
/// @param tree Complete rooted full binary tree.
/// @param leaf_states Node-indexed states; internal entries ignored.
/// @return Exact score, actual masks and deterministic witness.
/// @throws std::invalid_argument On malformed tree, wrong state-vector length or leaf state outside0..31.
/// @throws std::length_error On nonrepresentable node/count geometry.
/// @par Example
/// @code
/// auto a=dm::algorithms::fitch_analyze({0,{{-1,-1}}},{31});
/// std::print("{} {}\n",a.score,a.states[0]); // 0 31
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
FitchAnalysis fitch_analyze(const ParsimonyTree& tree, const std::vector<int>& leaf_states);

/// @brief Capture every actual Fitch set write and witness edge assignment from one producer run.
/// @details Complete validation and logical admission precede copying or evaluation. The quota is7N for nonempty trees and one for empty trees, default one million; zero disables the optional cap. Iterative traversal avoids recursive call-stack exhaustion. Captures the actual scoring/witness passes without replaying a second Fitch implementation.
/// @param tree Complete rooted full binary tree.
/// @param leaf_states Node-indexed states, with ignored internal entries.
/// @param max_entries Complete logical cap, zero off.
/// @return Owned inputs, actual node/edge histories and same-run analysis.
/// @throws std::invalid_argument On malformed tree, wrong state-vector length or leaf state outside0..31.
/// @throws std::length_error On nonrepresentable node/count geometry.
/// @throws std::length_error If complete logical admission exceeds its cap.
/// @par Example
/// @code
/// auto t=dm::algorithms::fitch_trace({0,{{-1,-1}}},{0},7);
/// std::print("{}\n",t.steps.size()); // 1
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
FitchTrace fitch_trace(
    const ParsimonyTree&    tree,
    const std::vector<int>& leaf_states,
    std::size_t             max_entries = 1000000);

/// @brief Return the exact one-character score from the canonical checked producer.
/// @details Same constraints and iterative producer as fitch_analyze; no topology search.
/// @param tree Complete rooted full binary tree.
/// @param leaf_state Node-indexed states0..31 at leaves.
/// @return Exact minimum unordered unit-change count.
/// @throws std::invalid_argument On malformed tree, wrong state-vector length or leaf state outside0..31.
/// @throws std::length_error On nonrepresentable node/count geometry.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fitch_small_parsimony({0,{{-1,-1}}},{0})); // 0
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
int fitch_small_parsimony(const ParsimonyTree& tree, const std::vector<int>& leaf_state);

/// @brief Return actual exact scores for all supplied character columns.
/// @details Validate the complete tree and every column before evaluating any site. Reuse the checked structural traversal order; each site runs the same canonical producer. Empty column collection still validates the tree. O(N*(C+1)) time including validation, O(N+C) working/result scalar space excluding supplied columns.
/// @param tree Complete rooted full binary tree.
/// @param leaf_states Columns, each with exactly N node-indexed entries.
/// @return One exact score per original column, in order.
/// @throws std::invalid_argument On malformed tree, wrong state-vector length or leaf state outside0..31.
/// @throws std::length_error On nonrepresentable node/count geometry.
/// @par Example
/// @code
/// auto scores=dm::algorithms::parsimony_site_scores({0,{{-1,-1}}},{{0},{1}});
/// std::print("{}\n",scores.size()); // 2
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
std::vector<int> parsimony_site_scores(
    const ParsimonyTree&                 tree,
    const std::vector<std::vector<int>>& leaf_states);

/// @brief Sum canonical per-site scores with checked int accumulation.
/// @details Sites have independent states and equal unit edge costs; sum is the additive multi-character objective for this fixed topology.
/// @param tree Complete rooted full binary tree.
/// @param leaf_states Complete node-indexed character columns.
/// @return Exact representable sum of actual site scores.
/// @throws std::invalid_argument On malformed tree, wrong state-vector length or leaf state outside0..31.
/// @throws std::length_error On nonrepresentable node/count geometry.
/// @throws std::overflow_error If the complete summed score exceeds INT_MAX.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::parsimony_score({0,{{-1,-1}}},{{0},{1}})); // 0
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
int parsimony_score(const ParsimonyTree& tree, const std::vector<std::vector<int>>& leaf_states);

/// @brief Build the original four-taxon topology ((a,b),(c,d)).
/// @details Leaves0..3, internals4 and5, root6; supplied ids must be a permutation of0..3.
/// @param a First left-pair leaf.
/// @param b Second left-pair leaf.
/// @param c First right-pair leaf.
/// @param d Second right-pair leaf.
/// @return Owned seven-node original-style tree.
/// @throws std::invalid_argument If ids are not a permutation of0..3.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::parsimony_four_taxon_tree(0,1,2,3).root); // 6
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
ParsimonyTree parsimony_four_taxon_tree(int a, int b, int c, int d);

/// @brief Map four valid leaf states to the original seven-node column layout.
/// @details Internal input entries are -1 and ignored by the producer.
/// @param states Four leaf states0..31 in taxon-id order.
/// @return Owned seven-entry original-style column.
/// @throws std::invalid_argument If a leaf state is outside0..31.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::parsimony_four_taxon_column({0,0,1,1}).size()); // 7
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
std::vector<int> parsimony_four_taxon_column(std::array<int, 4> states);

/// @brief Retain the original five character columns across four taxa.
/// @details Leaf rows are0011,0011,1100,0100,0001; each maps to the original seven-node layout.
/// @return Owned original five columns.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::parsimony_demo_columns().size()); // 5
/// @endcode
/// @see fitch_analyze, fitch_trace
/// @par Reference
/// Fitch (1971), Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology, Systematic Zoology20(4),406–416, doi:10.2307/2412116.
std::vector<std::vector<int>> parsimony_demo_columns();

} // namespace datamunge::algorithms
