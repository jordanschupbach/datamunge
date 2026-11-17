#pragma once
/// @file transitive_closure.hpp
/// @brief Actual checked Warshall phases and reachability updates.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own ClosureOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ClosureOptions r{};
/// (void)r;
/// @endcode
struct ClosureOptions {
  std::size_t max_cells{1000000}; ///< Complete n*n bit geometry; zero disables.
  std::size_t max_updates{
      200000000}; ///< Actual executed inner candidate updates, including no changes; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual phase records plus copied bit values; zero disables.
};
/// @brief Own ClosurePhase values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ClosurePhase r{};
/// (void)r;
/// @endcode
struct ClosurePhase {
  std::size_t intermediate{}; ///< Actual newly allowed intermediate vertex k.
  std::size_t
      candidate_updates{}; ///< Actual cumulative executed inner j updates through this phase.
  std::size_t new_paths{}; ///< Actual cumulative false-to-true updates through this phase.
  std::vector<std::vector<bool>> after; ///< Actual complete reachability after processing this k.
};
/// @brief Own ClosureRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ClosureRun r{};
/// (void)r;
/// @endcode
struct ClosureRun {
  std::vector<std::vector<bool>>
      reachable; ///< Actual final directed reachability; diagonal includes cycles even if reflexivefalse.
  std::size_t seeded_edges{};      ///< Actual input edge assignments, including duplicates.
  std::size_t row_tests{};         ///< Actual r[i][k] outer predicate checks, n*n.
  std::size_t candidate_updates{}; ///< Actual executed inner j updates when r[i][k] is true.
  std::size_t
      new_paths{}; ///< Actual false-to-true inner updates, excluding seeds/reflexive diagonal.
  std::vector<ClosurePhase>
      phases; ///< Optional actual post-k matrices, not a second closure replay.
};
/// @brief Compute directed reachability through the actual checked Warshall kernel.
/// Validate every directed endpoint before matrix allocation, then seed all edges and optionally the diagonal. For k,i test current R[i][k]; when true visit every j and assign R[i][j]=R[i][j] or R[k][j]. After k, paths using intermediate vertices0..k are represented. Reflexivefalse means positive-length reachability, not a forced-false diagonal: directed cycles still reach themselves. Empty graph is valid; loops/duplicates allowed and seeded assignment count includes duplicates. Worst-case cubic updates and quadratic bit storage; optional complete phase copies add cubic history. Complete matrix and actual update/capture policies are event/value counts, not wall time/peak bytes. Owned results, no shared state.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed endpoint pairs.
/// @param reflexive Seed zero-length self paths when true; defaulttrue.
/// @param options Complete matrix and actual work/history policies; zero disables.
/// @return Owned actual reachable matrix, selected counts and optional post-k phases.
/// @throws std::invalid_argument Complete edge endpoint outside0..n-1.
/// @throws std::length_error Complete matrix, actual updates/history or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::transitive_closure_run(3,{{0,1},{1,2}});
/// std::print("{}\n",r.reachable[0][2]); // true
/// @endcode
/// @par References
/// - Warshall, A Theorem on Boolean Matrices, JACM9 (1962),11-12.
/// @see transitive_closure()
ClosureRun transitive_closure_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive = true,
    const ClosureOptions&                                   options   = {});
/// @brief Compute directed reachability through the actual checked Warshall kernel.
/// Validate every directed endpoint before matrix allocation, then seed all edges and optionally the diagonal. For k,i test current R[i][k]; when true visit every j and assign R[i][j]=R[i][j] or R[k][j]. After k, paths using intermediate vertices0..k are represented. Reflexivefalse means positive-length reachability, not a forced-false diagonal: directed cycles still reach themselves. Empty graph is valid; loops/duplicates allowed and seeded assignment count includes duplicates. Worst-case cubic updates and quadratic bit storage; optional complete phase copies add cubic history. Complete matrix and actual update/capture policies are event/value counts, not wall time/peak bytes. Owned results, no shared state.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed endpoint pairs.
/// @param reflexive Seed zero-length self paths when true; defaulttrue.
/// @param options Complete matrix and actual work/history policies; zero disables.
/// @return Owned actual reachable matrix, selected counts and optional post-k phases.
/// @throws std::invalid_argument Complete edge endpoint outside0..n-1.
/// @throws std::length_error Complete matrix, actual updates/history or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::transitive_closure_trace(3,{{0,1},{1,2}});
/// std::print("{}\n",r.reachable[0][2]); // true
/// @endcode
/// @par References
/// - Warshall, A Theorem on Boolean Matrices, JACM9 (1962),11-12.
/// @see transitive_closure()
ClosureRun transitive_closure_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive = true,
    const ClosureOptions&                                   options   = {});
/// @brief Preserve checked directed Warshall reachability with default policies.
/// Validate every directed endpoint before matrix allocation, then seed all edges and optionally the diagonal. For k,i test current R[i][k]; when true visit every j and assign R[i][j]=R[i][j] or R[k][j]. After k, paths using intermediate vertices0..k are represented. Reflexivefalse means positive-length reachability, not a forced-false diagonal: directed cycles still reach themselves. Empty graph is valid; loops/duplicates allowed and seeded assignment count includes duplicates. Worst-case cubic updates and quadratic bit storage; optional complete phase copies add cubic history. Complete matrix and actual update/capture policies are event/value counts, not wall time/peak bytes. Owned results, no shared state.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed endpoint pairs.
/// @param reflexive Seed zero-length self paths when true; defaulttrue.
/// @return Owned actual reachability bit matrix.
/// @throws std::invalid_argument Invalid endpoint.
/// @throws std::length_error Default matrix/work or representational capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::transitive_closure(3,{{0,1},{1,2}});
/// std::print("{}\n",r[0][2]); // true
/// @endcode
/// @see transitive_closure_run()
std::vector<std::vector<bool>> transitive_closure(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive = true);
} // namespace datamunge::algorithms
