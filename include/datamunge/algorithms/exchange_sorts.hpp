#pragma once
/// @file exchange_sorts.hpp
/// @brief Integer adjacent-exchange and prefix-reversal sorting entry points.

#include <cstddef>
#include <datamunge/algorithms/cocktail_shaker_sort.hpp>
#include <datamunge/algorithms/gnome_sort.hpp>
#include <datamunge/algorithms/odd_even_sort.hpp>
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

/// @brief Owned sorted integers and actual prefix lengths reversed by pancake_sort().
/// Applying flips to the original input reproduces sorted; no optimal-flip claim.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pancake_sort({2,1}).flips); // [2]
/// @endcode
struct PancakeResult {
  std::vector<int>         sorted; ///< Nondecreasing integers, preserving multiplicities.
  std::vector<std::size_t> flips;  ///< Actual prefix lengths, each at least two.
};
/// @brief One actual greedy maximum search, including any resulting reversals.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pancake_sort_trace({2,1}).steps[0].maximum_index); // 0
/// @endcode
struct PancakeStep {
  std::size_t      prefix;        ///< Current unsorted prefix length.
  std::size_t      maximum_index; ///< First maximal value in that prefix before flips.
  std::size_t      comparisons;   ///< Actual value comparisons, prefix minus one.
  std::size_t      flip_begin;    ///< Begin index into result.flips.
  std::size_t      flip_end;      ///< Exclusive end of this search's actual flips.
  std::vector<int> values;        ///< Complete array immediately after this prefix is settled.
};
/// @brief Same-producer greedy decisions and post-reversal states, with owned output.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pancake_sort_trace({2,1}).result.sorted); // [1, 2]
/// @endcode
struct PancakeTrace {
  PancakeResult                 result; ///< Ordinary output from this actual run.
  std::vector<PancakeStep>      steps;  ///< Actual searches, including already-settled skips.
  std::vector<std::vector<int>> states; ///< Initial array followed by each actual reversal result.
};
/// @brief Greedily sort integers using only prefix reversals.
/// First maximum wins ties; no stability guarantee. At most 2(n-1) reversals for
/// n>=1, O(n^2) comparisons and element swaps, O(n) owned output/flip storage.
/// Lvalue input is copied, never mutated. Empty/singleton returns without flips.
/// @param a Owned integers, including duplicates and signed extremes.
/// @return Sorted values and actual reversal lengths; not a shortest sequence.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pancake_sort({3,1,2}).sorted); // [1, 2, 3]
/// @endcode
/// @par References
/// - W. H. Gates and C. H. Papadimitriou (1979), Bounds for sorting by prefix
///   reversal, Discrete Mathematics 27, 47-57, doi:10.1016/0012-365X(79)90068-2.
PancakeResult pancake_sort(std::vector<int> a);
/// @brief Retain actual prefix searches and reversal snapshots from pancake_sort's kernel.
/// O(n^2) history entries in the worst case. No replay or second sort is used.
/// @param a Owned input, unchanged for lvalue callers.
/// @param max_entries Logical retained scalar cap: initial values, each step's
/// full values plus one record, and each reversal snapshot; default one million,
/// zero disables policy. Does not count ordinary result storage or peak bytes.
/// @return Owned result, actual searches and initial/post-reversal states.
/// @throws std::length_error If the history cap or representable storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pancake_sort_trace({2,1}).states.size()); // 2
/// @endcode
PancakeTrace pancake_sort_trace(std::vector<int> a, std::size_t max_entries = 1000000);

} // namespace datamunge::algorithms
