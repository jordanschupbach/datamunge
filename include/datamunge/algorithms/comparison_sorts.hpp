#pragma once

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

/// @brief *Comb sort* (Dobosiewicz 1980; Lacey & Box 1991): generalizes bubble sort the way Shell
///        sort generalizes insertion sort. It compares and swaps elements a *gap* apart, shrinking
///        the gap each pass by the empirically good factor 1.3, until the gap is 1 and a final pass
///        makes no swaps. The large early gaps kill "turtles" -- small values stranded near the end
///        that bubble sort moves only one step per pass -- lifting the typical cost from bubble
///        sort's @c O(n^2) to roughly @c O(n log n) in practice. It is in-place and not stable.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> comb_sort(std::vector<int> a);

} // namespace datamunge::algorithms
