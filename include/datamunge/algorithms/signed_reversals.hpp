#pragma once
/// @file signed_reversals.hpp
/// @brief Checked exact signed-reversal BFS and actual predecessor path.
#include <cstddef>
#include <utility>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief SignedReversalResult stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SignedReversalResult value{};
/// std::print("{}\n",value.distance); // 0
/// @endcode
struct SignedReversalResult {
  int distance{};                             ///< Proven minimum reversal count when BFS completes.
  std::vector<std::pair<int, int>> reversals; ///< Actual selected inclusive zero-based reversals.
};
/// @brief SignedReversalOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SignedReversalOptions value{};
/// std::print("{}\n",value.max_genes); // 8
/// @endcode
struct SignedReversalOptions {
  std::size_t max_genes{
      8}; ///< Complete gene count; zero disables policy but int indexing remains bounded.
  std::size_t max_states{
      1000000}; ///< Actual distinct visited states including root; zero disables.
  std::size_t max_edges{20000000}; ///< Actual generated reversal neighbors; zero disables.
};
/// @brief SignedReversalState stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SignedReversalState value{};
/// std::print("{}\n",value.breakpoints); // 0
/// @endcode
struct SignedReversalState {
  std::vector<int> permutation; ///< Actual selected predecessor-chain state.
  int breakpoints{}; ///< Actual framed nonconsecutive adjacencies; ceil(b/2) is a distance lower bound.
  std::pair<int, int> reversal{-1, -1}; ///< Actual producing edge; {-1,-1} at root.
};
/// @brief SignedReversalLayer stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SignedReversalLayer value{};
/// std::print("{}\n",value.depth); // 0
/// @endcode
struct SignedReversalLayer {
  int         depth{};      ///< Actual BFS expansion depth.
  std::size_t expanded{};   ///< Actual dequeued states at this depth.
  std::size_t generated{};  ///< Actual tested reversal neighbors at this depth.
  std::size_t discovered{}; ///< Actual new states at this depth, discovered at depth+1.
};
/// @brief SignedReversalRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SignedReversalRun value{};
/// std::print("{}\n",value.states); // 0
/// @endcode
struct SignedReversalRun {
  SignedReversalResult result;      ///< Proven minimum and actual selected predecessor edges.
  std::size_t          states{};    ///< Actual visited unique states including initial and target.
  std::size_t          generated{}; ///< Actual neighbor transformations attempted.
  std::vector<SignedReversalState>
      path; ///< Optional actual selected predecessor chain including root.
  std::vector<SignedReversalLayer> layers; ///< Actual BFS expansion work per visited depth.
};
/// @brief Find an exact shortest signed-reversal sequence.
/// Input must contain each absolute gene1..n exactly once. Empty permutation is identity. BFS visits reversals [i,j] in increasing i then j, storing each first predecessor; first discovered identity is optimal because every edge has unit cost. The reported path comes from those actual stored states, not a separate replay. n genes have at most 2^n*n! states; generating n(n+1)/2 O(n)-copy neighbors and ordered-map lookup per expanded state is exponential. Bounds abort with an exception, never an invented distance or optimality claim. No biological inference from synthetic toy permutations. Owned working storage, no shared state.
/// @param start Borrowed complete valid signed permutation.
/// @param options Complete size and actual visited-state/edge policies; zero disables policy.
/// @return Owned exact result, actual search counts/layers and optional selected path.
/// @throws std::invalid_argument Zero, duplicate or out-of-range absolute gene.
/// @throws std::length_error Input, state/edge policy, int indexing or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::signed_reversal_run({-2,-1});
/// std::print("{}\n",r.result.distance); // 1
/// @endcode
/// @par References
/// - Hannenhalli and Pevzner, Transforming cabbage into turnip, JACM 46(1), 1999, DOI 10.1145/300515.300516. This implementation uses exhaustive BFS, not their polynomial algorithm.
/// @see signed_breakpoints()
SignedReversalRun signed_reversal_run(
    const std::vector<int>&      start,
    const SignedReversalOptions& options = {});
/// @brief Find an exact shortest signed-reversal sequence with actual selected predecessor states.
/// Input must contain each absolute gene1..n exactly once. Empty permutation is identity. BFS visits reversals [i,j] in increasing i then j, storing each first predecessor; first discovered identity is optimal because every edge has unit cost. The reported path comes from those actual stored states, not a separate replay. n genes have at most 2^n*n! states; generating n(n+1)/2 O(n)-copy neighbors and ordered-map lookup per expanded state is exponential. Bounds abort with an exception, never an invented distance or optimality claim. No biological inference from synthetic toy permutations. Owned working storage, no shared state.
/// @param start Borrowed complete valid signed permutation.
/// @param options Complete size and actual visited-state/edge policies; zero disables policy.
/// @return Owned exact result, actual search counts/layers and optional selected path.
/// @throws std::invalid_argument Zero, duplicate or out-of-range absolute gene.
/// @throws std::length_error Input, state/edge policy, int indexing or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::signed_reversal_trace({-2,-1});
/// std::print("{}\n",r.result.distance); // 1
/// @endcode
/// @par References
/// - Hannenhalli and Pevzner, Transforming cabbage into turnip, JACM 46(1), 1999, DOI 10.1145/300515.300516. This implementation uses exhaustive BFS, not their polynomial algorithm.
/// @see signed_breakpoints()
SignedReversalRun signed_reversal_trace(
    const std::vector<int>&      start,
    const SignedReversalOptions& options = {});
/// @brief Return exact signed-reversal distance through the preserved API.
/// Checked BFS with default policies; throws rather than returning an unproved answer on exhaustion.
/// @param start Valid complete signed permutation.
/// @return Owned exact minimum and selected reversals.
/// @throws std::invalid_argument Malformed permutation.
/// @throws std::length_error Default search policy or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sort_by_reversals({-2,-1}).distance); // 1
/// @endcode
/// @see signed_reversal_run()
SignedReversalResult sort_by_reversals(const std::vector<int>& start);
/// @brief Count framed directed breakpoints in a signed permutation.
/// Frames with0 and n+1 and counts b!=a+1. A signed reversal preserves all internal directed adjacencies and changes at most two boundary adjacencies, so ceil(breakpoints/2), not the raw count, lower-bounds distance. Validates genes, supports empty input; O(n) time/storage, no shared state.
/// @param perm Borrowed complete valid signed permutation.
/// @return Nonnegative framed breakpoint count.
/// @throws std::invalid_argument Invalid genes.
/// @throws std::length_error Cannot represent framed indices in int.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::signed_breakpoints({-2,-1})); // 2
/// @endcode
/// @see signed_reversal_run()
int signed_breakpoints(const std::vector<int>& perm);
/// @brief Apply one checked inclusive signed reversal to an owned permutation.
/// Reverses the interval and negates each gene. Validates permutation and indices before mutation; O(n) validation and owned output, no shared state.
/// @param permutation Owned complete valid signed permutation.
/// @param first Inclusive zero-based first index.
/// @param last Inclusive zero-based final index, at least first.
/// @return Owned transformed permutation.
/// @throws std::invalid_argument Malformed permutation or interval.
/// @throws std::length_error Int indexing capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::signed_apply_reversal({-2,-1},0,1)); // [1, 2]
/// @endcode
/// @see signed_reversal_run()
std::vector<int> signed_apply_reversal(std::vector<int> permutation, int first, int last);
/// @brief Historical compatibility helper namespace.
namespace detail {
/// @brief Preserve the historical checked reversal helper.
/// Delegates to the canonical validated transformation.
/// @param p Owned valid signed permutation.
/// @param i First inclusive position.
/// @param j Last inclusive position.
/// @return Owned transformed sequence.
/// @throws std::invalid_argument Malformed genes/interval.
/// @throws std::length_error Index capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::detail::apply_reversal({-1},0,0)); // [1]
/// @endcode
/// @see signed_apply_reversal()
inline std::vector<int> apply_reversal(std::vector<int> p, int i, int j) {
  return signed_apply_reversal(std::move(p), i, j);
}
} // namespace detail
} // namespace datamunge::algorithms
