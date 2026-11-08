#pragma once
/// @file comparison_sorts.hpp
/// @brief Integer insertion, selection, Shell and comb sorting entry points.

#include <datamunge/algorithms/comb_sort.hpp>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Insertion sort*: builds the sorted result one element at a time, growing a sorted prefix
///        and inserting each new element into its place by shifting larger elements right. It is
///        simple, in-place, and *stable*, and -- crucially -- *adaptive*: on already- or
///        nearly-sorted input it runs in @c O(n) (each new element barely moves), while the worst
///        case (reverse-sorted) is @c O(n^2). It is the method of choice for small or almost-sorted
///        arrays, and the workhorse inside hybrid sorts like Timsort and Introsort at small sizes.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> insertion_sort(std::vector<int> a);

/// @brief *Selection sort*: repeatedly selects the smallest remaining element and appends it to the
///        sorted prefix by a single swap. It always makes exactly @c n-1 swaps -- the *fewest writes*
///        of any simple sort, valuable when writes are expensive -- but always @c O(n^2) comparisons,
///        with no adaptivity: sorted and random inputs cost the same. It is in-place and not stable.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> selection_sort(std::vector<int> a);

/// @brief *Shell sort* (Shell, 1959): generalizes insertion sort by first sorting elements that are
///        far apart (a distance called the *gap*), then progressively shrinking the gap to 1. Early
///        long-range passes move elements most of the way home cheaply, so the final gap-1 insertion
///        pass has little left to do, breaking the @c O(n^2) barrier: with the simple halving gap
///        sequence used here it runs in roughly @c O(n^2) worst case but @c O(n^{3/2}) on average,
///        and better gap sequences do better still. It is in-place and not stable.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> shell_sort(std::vector<int> a);

/// @brief Return a nondecreasing integer copy using the plain 10/13 comb-gap schedule.
/// Same executing kernel as comb_sort_work()/comb_sort_trace(), with counters
/// and snapshots disabled. Strict nonadjacent swaps are not stable. Initial
/// gap=n is shrunk before each pass by exact max(1,floor(10*gap/13)), without
/// overflowing gap multiplication or applying a special gap adjustment. Sorting
/// stops only after a zero-swap gap-one scan. Empty/singleton inputs return
/// immediately. Worst-case upper bound O(n^2), sorted-input Theta(n log n);
/// a measured random curve is not a universal O(n log n) guarantee. The owned
/// input buffer is sorted with constant auxiliary state; lvalue callers keep
/// their vector. Full signed integer range, no shared state or extra allocation.
///
/// @param a Owned integer input; an lvalue is copied by the call.
/// @return Owned sorted values preserving multiplicities.
/// @throws std::length_error If requested vector storage exceeds capacity.
/// @throws std::bad_alloc If the by-value input copy cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::comb_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @par References
/// - W. Dobosiewicz (1980), An efficient variation of bubble sort, Information
///   Processing Letters 11(1), 5-6, doi:10.1016/0020-0190(80)90022-8 (historical variant).
/// - P. E. Black (2022), Comb sort, https://xlinux.nist.gov/dads/HTML/combSort.html.
/// @see comb_sort_work(), comb_sort_trace(), comb_sort_next_gap()
std::vector<int> comb_sort(std::vector<int> a);

} // namespace datamunge::algorithms
