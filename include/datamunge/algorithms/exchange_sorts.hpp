#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Gnome sort* (Hamid Sarbazi-Azad, 2000): the simplest sort to describe -- like a garden
///        gnome ordering flower pots. Walk forward; if the current pair is in order step forward,
///        otherwise swap them and step *back*. That single rule sorts the array with one loop, no
///        nested loops and no bounds bookkeeping. It behaves like insertion sort -- @c O(n^2) worst
///        case, @c O(n) on nearly-sorted input -- and is in-place and stable.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> gnome_sort(std::vector<int> a);

/// @brief *Cocktail shaker sort* (bidirectional bubble sort): a bubble sort that alternates
///        direction each pass -- left to right bubbling the largest element to the end, then right
///        to left bubbling the smallest to the front. Sweeping both ways fixes the "turtles" (small
///        values near the end) that plain bubble sort moves only one step per pass, roughly halving
///        the passes, though it remains @c O(n^2). It is in-place and stable, and stops early once a
///        pass makes no swaps.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
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
