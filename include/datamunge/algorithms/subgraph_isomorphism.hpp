#pragma once
/// @file subgraph_isomorphism.hpp
/// @brief Checked non-induced injective undirected matching with actual pruning states.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own SubgraphMatch for checked non-induced matching.
/// Fields describe actual normalized geometry, producing decisions or invocation-local policies. Owned containers, no retained caller storage or shared mutable state. Construction alone does not execute matching.
/// @par Example
/// @code
/// dm::algorithms::SubgraphMatch r{};
/// (void)r;
/// @endcode
/// @see subgraph_isomorphism_run()
struct SubgraphMatch {
  bool found{false}; ///< Actual complete adjacency-preserving injection was found.
  std::vector<int>
      mapping; ///< Actual target index per pattern index, empty on failure or empty-pattern success.
};
/// @brief Own SubgraphOptions for checked non-induced matching.
/// Fields describe actual normalized geometry, producing decisions or invocation-local policies. Owned containers, no retained caller storage or shared mutable state. Construction alone does not execute matching.
/// @par Example
/// @code
/// dm::algorithms::SubgraphOptions r{};
/// (void)r;
/// @endcode
/// @see subgraph_isomorphism_run()
struct SubgraphOptions {
  std::size_t max_matrix_cells{
      1000000}; ///< Complete pattern-square plus target-square geometry; zero disables.
  std::size_t max_pattern_vertices{
      512}; ///< Complete pattern order; zero disables policy, intrinsic1024 remains.
  std::size_t max_candidate_trials{
      20000000}; ///< Actual target candidates inspected, including used/degree prunes; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual event records plus copied mapping entries and captured matrices/degrees; zero disables.
};
/// @brief Own SubgraphStep for checked non-induced matching.
/// Fields describe actual normalized geometry, producing decisions or invocation-local policies. Owned containers, no retained caller storage or shared mutable state. Construction alone does not execute matching.
/// @par Example
/// @code
/// dm::algorithms::SubgraphStep r{};
/// (void)r;
/// @endcode
/// @see subgraph_isomorphism_run()
struct SubgraphStep {
  std::size_t depth{};    ///< Actual next pattern position/recursion depth.
  int         target{-1}; ///< Actual candidate, -1 for completed success.
  std::string
      action; ///< Actual used-prune, degree-prune, edge-prune, assign, backtrack or success.
  std::vector<int>
      mapping; ///< Actual current complete mapping after this event, with-1 for unassigned.
};
/// @brief Own SubgraphRun for checked non-induced matching.
/// Fields describe actual normalized geometry, producing decisions or invocation-local policies. Owned containers, no retained caller storage or shared mutable state. Construction alone does not execute matching.
/// @par Example
/// @code
/// dm::algorithms::SubgraphRun r{};
/// (void)r;
/// @endcode
/// @see subgraph_isomorphism_run()
struct SubgraphRun {
  SubgraphMatch match; ///< Actual first complete injection, increasing pattern and target indices.
  std::vector<bool>
      target_matched; ///< Actual target indices used in selected successful mapping, all false on failure.
  std::vector<int>
      pattern_degrees; ///< Actual degrees after symmetrizing lists, removing duplicate edges and ignoring loops.
  std::vector<int> target_degrees;     ///< Actual normalized target degrees.
  std::size_t      candidate_trials{}; ///< Actual inspected target candidates.
  std::size_t      used_prunes{};      ///< Actual candidates already assigned.
  std::size_t      degree_prunes{};    ///< Actual unused candidates with insufficient degree.
  std::size_t
      assigned_pair_visits{}; ///< Actual previously mapped pattern positions examined, including nonedges.
  std::size_t
      required_edge_tests{}; ///< Actual normalized target adjacency reads for required pattern edges.
  std::size_t adjacency_prunes{}; ///< Actual candidates rejected at first missing required edge.
  std::size_t assignments{};      ///< Actual new candidate assignments.
  std::size_t backtracks{};       ///< Actual assignments undone after failed recursive children.
  std::vector<std::vector<char>>
      pattern_matrix; ///< Optional actual normalized pattern adjacency, zero/one bytes.
  std::vector<std::vector<char>>
      target_matrix;               ///< Optional actual normalized target adjacency, zero/one bytes.
  std::vector<SubgraphStep> steps; ///< Optional actual decisions/mappings in execution order.
};
/// @brief Find the first non-induced injective undirected subgraph match.
/// Counts must be nonnegative and adjacency list row counts must match exactly; every neighbor id validated before any early impossible-size return. Normalize to symmetric simple adjacency by union of listed directions; ignore loops and duplicates. Every pattern edge must map to a target edge; pattern nonedges are unconstrained, so this is graph monomorphism, not induced matching. Assign pattern vertices in index order and target candidates in increasing order; prune used targets, insufficient normalized degree and missing edges to already placed vertices. Empty pattern succeeds with empty mapping. Intrinsic pattern depth1024 protects recursion; complete dense geometry checked before allocation. Worst-case injective assignment search exponential, matrix storage quadratic; optional real mapping snapshots add actual search-sized storage. Input borrowed only during call, outputs own values and no shared state.
/// @param np Complete nonnegative pattern order.
/// @param padj Complete borrowed pattern adjacency rows.
/// @param ng Complete nonnegative target order.
/// @param gadj Complete borrowed target adjacency rows.
/// @param options Complete matrix/depth and actual search/capture policies; zero disables policies except intrinsic depth.
/// @return Owned actual first match, degree/pruning counters, selected-target flags and optional normalized matrices/events.
/// @throws std::invalid_argument Negative order, row-count mismatch or out-of-range neighbor id.
/// @throws std::length_error Complete geometry/depth, actual trial/history or counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::subgraph_isomorphism_run(1,{{}},1,{{}});
/// std::print("{} {}\n",r.match.found,r.match.mapping); // true [0]
/// @endcode
/// @par References
/// - J. R. Ullmann, An Algorithm for Subgraph Isomorphism, JACM23(1),1976, DOI10.1145/321921.321925. This simpler index-ordered degree/assigned-edge backtracker does not implement Ullmann matrix refinement.
/// @see subgraph_isomorphism()
SubgraphRun subgraph_isomorphism_run(
    int                                  np,
    const std::vector<std::vector<int>>& padj,
    int                                  ng,
    const std::vector<std::vector<int>>& gadj,
    const SubgraphOptions&               options = {});
/// @brief Find the first non-induced injective undirected subgraph match with actual pruning states.
/// Counts must be nonnegative and adjacency list row counts must match exactly; every neighbor id validated before any early impossible-size return. Normalize to symmetric simple adjacency by union of listed directions; ignore loops and duplicates. Every pattern edge must map to a target edge; pattern nonedges are unconstrained, so this is graph monomorphism, not induced matching. Assign pattern vertices in index order and target candidates in increasing order; prune used targets, insufficient normalized degree and missing edges to already placed vertices. Empty pattern succeeds with empty mapping. Intrinsic pattern depth1024 protects recursion; complete dense geometry checked before allocation. Worst-case injective assignment search exponential, matrix storage quadratic; optional real mapping snapshots add actual search-sized storage. Input borrowed only during call, outputs own values and no shared state.
/// @param np Complete nonnegative pattern order.
/// @param padj Complete borrowed pattern adjacency rows.
/// @param ng Complete nonnegative target order.
/// @param gadj Complete borrowed target adjacency rows.
/// @param options Complete matrix/depth and actual search/capture policies; zero disables policies except intrinsic depth.
/// @return Owned actual first match, degree/pruning counters, selected-target flags and optional normalized matrices/events.
/// @throws std::invalid_argument Negative order, row-count mismatch or out-of-range neighbor id.
/// @throws std::length_error Complete geometry/depth, actual trial/history or counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::subgraph_isomorphism_trace(1,{{}},1,{{}});
/// std::print("{} {}\n",r.match.found,r.match.mapping); // true [0]
/// @endcode
/// @par References
/// - J. R. Ullmann, An Algorithm for Subgraph Isomorphism, JACM23(1),1976, DOI10.1145/321921.321925. This simpler index-ordered degree/assigned-edge backtracker does not implement Ullmann matrix refinement.
/// @see subgraph_isomorphism()
SubgraphRun subgraph_isomorphism_trace(
    int                                  np,
    const std::vector<std::vector<int>>& padj,
    int                                  ng,
    const std::vector<std::vector<int>>& gadj,
    const SubgraphOptions&               options = {});
/// @brief Return the preserved checked non-induced subgraph match.
/// Same actual index-ordered producer with default policies. Validates malformed inputs instead of unchecked indexing. An empty pattern is a successful empty injection.
/// @param np Nonnegative pattern count.
/// @param padj Complete valid pattern adjacency rows.
/// @param ng Nonnegative target count.
/// @param gadj Complete valid target adjacency rows.
/// @return Owned first adjacency-preserving injection or found=false.
/// @throws std::invalid_argument Malformed graph.
/// @throws std::length_error Default geometry/work/depth/capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::subgraph_isomorphism(1,{{}},1,{{}});
/// std::print("{} {}\n",r.found,r.mapping[0]); // true 0
/// @endcode
/// @see subgraph_isomorphism_run()
SubgraphMatch subgraph_isomorphism(
    int                                  np,
    const std::vector<std::vector<int>>& padj,
    int                                  ng,
    const std::vector<std::vector<int>>& gadj);
/// @brief Generate a controlled simple undirected cycle fixture.
/// Returns both directions for each edge, in vertex order; no matching occurs. Owned structural data, no shared state.
/// @param n Cycle order in[3,100000].
/// @return Owned complete cycle adjacency lists.
/// @throws std::invalid_argument n below3.
/// @throws std::length_error n above100000.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::subgraph_cycle_graph(4)[0].size()); // 2
/// @endcode
/// @see subgraph_isomorphism_run()
std::vector<std::vector<int>> subgraph_cycle_graph(std::size_t n);
} // namespace datamunge::algorithms
