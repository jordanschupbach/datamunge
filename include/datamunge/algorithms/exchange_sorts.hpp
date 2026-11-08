#pragma once
/// @file exchange_sorts.hpp
/// @brief Integer adjacent-exchange and prefix-reversal sorting entry points.

#include <datamunge/algorithms/cocktail_shaker_sort.hpp>
#include <datamunge/algorithms/gnome_sort.hpp>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {



/// @brief Return a stable nondecreasing integer copy using alternating adjacent exchanges.
/// Forward passes fix the rightmost remaining maximum, backward passes the
/// leftmost minimum. Strict greater-than comparisons preserve equal-key order;
/// signed extremes are valid. Stops at the first zero-swap directional pass.
/// Each successful pass shrinks one bound by one: reversing direction does not
/// guarantee half the directional passes or comparisons of bubble sort.
/// Empty/singleton inputs return immediately. Worst-case time O(n^2), best O(n);
/// constant working state beyond the by-value input/output buffer. Lvalue callers
/// retain their original vector; no shared state is used. See the trace API for
/// actual pass states, identity labels and work counters from this same kernel.
///
/// @param a Owned integer input; an lvalue is copied by the call.
/// @return Owned sorted values, preserving multiplicities.
/// @throws std::bad_alloc If the by-value copy cannot be allocated.
/// @throws std::length_error If requested vector storage exceeds capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cocktail_shaker_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @par References
/// - P. E. Black and B. Bockholt, Bidirectional bubble sort, NIST Dictionary of
///   Algorithms and Data Structures, https://xlinux.nist.gov/dads/HTML/bidirectionalBubbleSort.html.
/// @see cocktail_shaker_sort_trace()
std::vector<int> cocktail_shaker_sort(std::vector<int> a);

/// @brief *Odd-even sort* (brick sort): repeatedly compares and swaps all (odd, even)-indexed
///        adjacent pairs, then all (even, odd) pairs, until a full round makes no swap. Each phase's
///        comparisons are independent, so it is a classic *parallel* sorting network -- every pair
///        in a phase can be compared at once. Serially it is @c O(n^2); on parallel hardware it runs
///        in @c O(n) phases. In-place and stable.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> odd_even_sort(std::vector<int> a);

/// @brief The result of @ref pancake_sort: the @ref sorted array and the sequence of @ref flips that
///        produced it. Each flip value @c k means "reverse the prefix of length @c k"; applying the
///        flips in order to the original array yields the sorted array.
struct PancakeResult {
    std::vector<int>         sorted; ///< the sorted array.
    std::vector<std::size_t> flips;  ///< prefix lengths reversed, in order.
};

/// @brief *Pancake sort*: sort using only *prefix reversals* ("flips"), like sorting a stack of
///        pancakes with a spatula. Repeatedly find the largest unsorted element, flip the prefix
///        that ends at it to bring it to the top, then flip the whole unsorted prefix to drop it
///        into its final place at the bottom of the unsorted region. This uses at most @c 2(n-1)
///        flips -- the metric of interest, since here the number of *reversals*, not comparisons,
///        is what counts (it models problems where only whole-prefix reversal is available). Runs in
///        @c O(n^2) comparisons.
///
/// @param a the input values.
/// @return the @ref PancakeResult with the sorted array and the flip sequence.
PancakeResult pancake_sort(std::vector<int> a);

} // namespace datamunge::algorithms
