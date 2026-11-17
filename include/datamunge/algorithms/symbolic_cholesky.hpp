#pragma once
/// @file symbolic_cholesky.hpp
/// @brief Actual structural fill under natural-order symmetric graph elimination.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SymbolicCholeskyOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SymbolicCholeskyOptions r{};
/// (void)r;
/// @endcode
/// @see symbolic_cholesky_run()
struct SymbolicCholeskyOptions {
  std::size_t max_vertices{1000000}; ///< Complete input vertices; zero disables.
  std::size_t max_edges{
      2000000}; ///< Complete normalized undirected input edges plus actual new fill edges; zero disables.
  std::size_t max_pair_tests{
      20000000}; ///< Actual higher-neighbor pair insertion attempts, including existing edges; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual captured records, neighbor entries and new-fill endpoint entries; zero disables.
};
/// @brief Own the SymbolicCholeskyStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SymbolicCholeskyStep r{};
/// (void)r;
/// @endcode
/// @see symbolic_cholesky_run()
struct SymbolicCholeskyStep {
  int              column{}; ///< Actual naturally ordered eliminated column.
  std::vector<int> higher;   ///< Actual current neighbors greater than column, in ascending order.
  std::vector<std::pair<int, int>>
              fill; ///< Actual previously absent undirected edges inserted at this column.
  std::size_t cumulative_pairs{}; ///< Actual cumulative higher-neighbor pair attempts.
};
/// @brief Own the SymbolicCholeskyRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SymbolicCholeskyRun r{};
/// (void)r;
/// @endcode
/// @see symbolic_cholesky_run()
struct SymbolicCholeskyRun {
  std::vector<std::vector<int>>
      lower; ///< Actual structural lower rows, only columns j<i; diagonal excluded.
  std::vector<std::pair<int, int>>
              fill;             ///< Actual newly inserted undirected edges, in producing order.
  std::size_t input_edges{};    ///< Actual normalized undirected input edges, loops ignored.
  std::size_t pair_tests{};     ///< Actual higher-neighbor pair attempts.
  std::size_t factor_entries{}; ///< Actual off-diagonal structural entries emitted into lower.
  std::vector<SymbolicCholeskyStep> steps; ///< Optional actual elimination neighbors and new edges.
};
/// @brief Predict actual structural Cholesky fill by natural-order elimination.
/// Consume a graph pattern only; every neighbor must be a valid int vertex ID. Normalize to an undirected simple graph using union of listed directions, ignoring loops/duplicates. Eliminate columns0..n-1 in natural order. Emit current higher neighbors as structural lower-factor entries, then connect every pair, recording actual new fill edges. This predicts potential structural nonzeros for a generic SPD matrix with that pattern; exact numerical cancellation may remove entries, and this API neither checks positive definiteness nor computes numerical factors. No permutation/ordering optimization is applied. Ordered sets give logarithmic insertion cost; pair attempts can be cubic in n with quadratic fill storage. Complete input and actual fill/pair/history policies are separate; owned output, no shared state.
/// @param adjacency Complete borrowed matrix pattern rows, neighbor IDs0..n-1.
/// @param options Complete vertices and actual normalized-edge/pair/capture budgets; zero disables.
/// @return Owned structural lower rows, actual new fill edges/counts and optional elimination states.
/// @throws std::invalid_argument Invalid neighbor ID.
/// @throws std::length_error Int vertex representation, complete vertices or actual fill/pair/history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::symbolic_cholesky_run({{1,2},{0},{0}});
/// std::print("{}\n",r.fill); // [(1, 2)]
/// @endcode
/// @par References
/// - Rose, Triangulated graphs and the elimination process, Journal of Mathematical Analysis and Applications32 (1970),597-609.
/// @see symbolic_cholesky()
SymbolicCholeskyRun symbolic_cholesky_run(
    const std::vector<std::vector<int>>& adjacency,
    const SymbolicCholeskyOptions&       options = {});
/// @brief Predict actual structural Cholesky fill by natural-order elimination.
/// Consume a graph pattern only; every neighbor must be a valid int vertex ID. Normalize to an undirected simple graph using union of listed directions, ignoring loops/duplicates. Eliminate columns0..n-1 in natural order. Emit current higher neighbors as structural lower-factor entries, then connect every pair, recording actual new fill edges. This predicts potential structural nonzeros for a generic SPD matrix with that pattern; exact numerical cancellation may remove entries, and this API neither checks positive definiteness nor computes numerical factors. No permutation/ordering optimization is applied. Ordered sets give logarithmic insertion cost; pair attempts can be cubic in n with quadratic fill storage. Complete input and actual fill/pair/history policies are separate; owned output, no shared state.
/// @param adjacency Complete borrowed matrix pattern rows, neighbor IDs0..n-1.
/// @param options Complete vertices and actual normalized-edge/pair/capture budgets; zero disables.
/// @return Owned structural lower rows, actual new fill edges/counts and optional elimination states.
/// @throws std::invalid_argument Invalid neighbor ID.
/// @throws std::length_error Int vertex representation, complete vertices or actual fill/pair/history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::symbolic_cholesky_trace({{1,2},{0},{0}});
/// std::print("{}\n",r.fill); // [(1, 2)]
/// @endcode
/// @par References
/// - Rose, Triangulated graphs and the elimination process, Journal of Mathematical Analysis and Applications32 (1970),597-609.
/// @see symbolic_cholesky()
SymbolicCholeskyRun symbolic_cholesky_trace(
    const std::vector<std::vector<int>>& adjacency,
    const SymbolicCholeskyOptions&       options = {});
/// @brief Preserve structural lower-row prediction using checked natural-order fill.
/// Consume a graph pattern only; every neighbor must be a valid int vertex ID. Normalize to an undirected simple graph using union of listed directions, ignoring loops/duplicates. Eliminate columns0..n-1 in natural order. Emit current higher neighbors as structural lower-factor entries, then connect every pair, recording actual new fill edges. This predicts potential structural nonzeros for a generic SPD matrix with that pattern; exact numerical cancellation may remove entries, and this API neither checks positive definiteness nor computes numerical factors. No permutation/ordering optimization is applied. Ordered sets give logarithmic insertion cost; pair attempts can be cubic in n with quadratic fill storage. Complete input and actual fill/pair/history policies are separate; owned output, no shared state.
/// @param adjacency Borrowed complete finite graph pattern.
/// @return Owned ascending off-diagonal structural column IDs per row.
/// @throws std::invalid_argument Invalid neighbor.
/// @throws std::length_error Default input/fill/work or representational capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::symbolic_cholesky({{1},{0}});
/// std::print("{}\n",r); // [[], [0]]
/// @endcode
/// @see symbolic_cholesky_run()
std::vector<std::vector<int>> symbolic_cholesky(const std::vector<std::vector<int>>& adjacency);
/// @brief Generate the retained square nearest-neighbor grid pattern.
/// Row-major side by side vertices with horizontal/vertical undirected edges, neighbor insertion right then down at each cell. Side0 returns empty; side<=1000 prevents int/size overflow and bounds fixture allocation. Owned structural fixture, no shared state.
/// @param side Square side length, default5 for the original25 vertices.
/// @return Owned symmetric adjacency rows.
/// @throws std::length_error Side above1000.
/// @par Example
/// @code
/// auto g=dm::algorithms::symbolic_cholesky_grid();
/// std::print("{}\n",g.size()); // 25
/// @endcode
/// @see symbolic_cholesky_run()
std::vector<std::vector<int>> symbolic_cholesky_grid(std::size_t side = 5);
} // namespace datamunge::algorithms
