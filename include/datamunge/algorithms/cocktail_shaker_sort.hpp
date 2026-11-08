#pragma once
/// @file cocktail_shaker_sort.hpp
/// @brief Owned records from the executing bidirectional adjacent-exchange sort.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned state after one actual directional pass of cocktail shaker sort.
/// Bounds are inclusive before/after the pass. A forward pass with swaps lowers
/// hi by one; every backward pass raises lo by one, including a stopping pass.
/// A zero-swap forward pass keeps both bounds. next_lo>next_hi means an empty
/// remaining window. A final zero-comparison pass is retained, matching the
/// legacy kernel. Only strict adjacent inversions swap; equal keys keep identity.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cocktail_shaker_sort_trace({3,2,1});
/// std::print("{} {}\n",t.passes[0].values,t.passes[0].swaps); // [2, 1, 3] 2
/// @endcode
/// @see CocktailSortTrace, cocktail_shaker_sort_trace()
struct CocktailSortPass {
  bool forward{true}; ///< True for ascending pair indices, false for descending indices.
  std::size_t lo{0}; ///< Inclusive lower bound on entry to this pass.
  std::size_t hi{0}; ///< Inclusive upper bound on entry to this pass.
  std::size_t next_lo{0}; ///< Inclusive lower bound after the pass and its bound update.
  std::size_t next_hi{0}; ///< Inclusive upper bound after the pass and its bound update.
  std::size_t key_comparisons{0}; ///< Evaluations of left value > right value; excludes loop tests.
  std::size_t swaps{0}; ///< Actual adjacent exchanges, each removing one strict inversion.
  bool stopped{false}; ///< True exactly when this zero-swap pass terminates the run.
  std::vector<int> values; ///< Owned complete values after the pass.
  std::vector<std::size_t> original_indices; ///< Original positions carried by the actual exchanges.
};

/// @brief Owned input/output, stable original identities and actual directional passes.
/// Empty/singleton inputs have no passes and zero work. Larger inputs include
/// the final zero-swap pass even if its window has one item and no comparisons.
/// Counts exclude copying, index initialization and loop control. Swaps equal
/// strict inversions in the original input. No input references or shared state
/// survive. Snapshot storage is O(n*p), where p<=n is the actual pass count;
/// output-only sorting uses constant auxiliary state beyond its owned input copy.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cocktail_shaker_sort_trace({2,1,2});
/// std::print("{} {}\n",t.output,t.original_indices); // [1, 2, 2] [1, 0, 2]
/// @endcode
/// @see CocktailSortPass, cocktail_shaker_sort_trace()
struct CocktailSortTrace {
  std::vector<int> input; ///< Owned original values.
  std::vector<int> output; ///< Owned nondecreasing final values.
  std::vector<std::size_t> original_indices; ///< Original positions in final stable order.
  std::vector<CocktailSortPass> passes; ///< Every executed pass, including the terminating pass.
  std::size_t key_comparisons{0}; ///< Checked sum of actual pass key comparisons.
  std::size_t swaps{0}; ///< Checked sum of actual adjacent swaps.
};

/// @brief Sort integers and record the executing cocktail-shaker passes without replay.
/// Same strict comparisons, alternating directions, early exits and one-position
/// bound updates as cocktail_shaker_sort(). Signed extremes require no arithmetic
/// on values. Equal keys retain ascending original positions. The by-value input
/// leaves lvalue callers unchanged. All records own their data; concurrent calls
/// use independent state. Time O(n^2), best O(n), with O(n*p) snapshot storage.
/// The positive pass cap is checked before each pass, including an empty final
/// pass: exceeding it throws without returning a partial trace. No cap applies
/// to the output-only legacy call. Counts are checked before addition.
/// @param a Owned integer input; empty/singleton inputs execute no passes.
/// @param max_passes Positive online snapshot cap, default 1024; not a comparison budget.
/// @return Sorted values, stable identities, owned actual states and checked counters.
/// @throws std::invalid_argument If max_passes is zero, even for empty input.
/// @throws std::length_error If an executed pass exceeds the cap or storage capacity.
/// @throws std::overflow_error If a work counter cannot fit size_t.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cocktail_shaker_sort_trace({3,2,1},3);
/// std::print("{} {} {}\n",t.output,t.key_comparisons,t.swaps); // [1, 2, 3] 3 3
/// @endcode
/// @par References
/// - P. E. Black and B. Bockholt, Bidirectional bubble sort, NIST Dictionary of
///   Algorithms and Data Structures, https://xlinux.nist.gov/dads/HTML/bidirectionalBubbleSort.html.
///   This implementation shrinks by one position per swapping directional pass.
/// @see cocktail_shaker_sort(), CocktailSortTrace
CocktailSortTrace cocktail_shaker_sort_trace(std::vector<int> a,std::size_t max_passes=1024);

} // namespace datamunge::algorithms
