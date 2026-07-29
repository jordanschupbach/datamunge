#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Returns the @p k-th *smallest* element of @p data (0-indexed: k=0 is the minimum, k=1
///        the second smallest, ..., k=n-1 the maximum) -- the k-th *order statistic* -- without
///        fully sorting the array. This is Hoare's Quickselect (1961): pick a pivot, *partition*
///        the array so that everything smaller precedes it and everything larger follows it, note
///        the pivot's final rank p, and then recurse into *only one* side -- the left if k < p, the
///        right if k > p, and stop if k == p. Because each step discards one partition instead of
///        recursing into both (as quicksort does), the expected work follows the recurrence
///        T(n) = T(n/2) + O(n) = O(n) on random input, versus O(n log n) to sort the whole array
///        just to read one element. This overload takes @p data *by value* precisely because it
///        reorders it internally; the caller's array is untouched. The pivot is chosen by the
///        *median-of-three* rule (median of the first, middle, and last elements), a cheap
///        heuristic that avoids the worst case on already-sorted and reverse-sorted inputs, though
///        a truly adversarial permutation can still force the O(n^2) worst case -- for a guaranteed
///        linear bound use @ref quickselect_mom. Duplicate values are handled correctly.
///
/// @param data the values to select from (taken by value; reordered internally, caller's copy is
///        unaffected).
/// @param k the 0-indexed rank to retrieve (0 = minimum, data.size()-1 = maximum).
/// @return the k-th smallest element of @p data.
/// @throws std::invalid_argument if @p data is empty or if k >= data.size().
[[nodiscard]] std::int64_t quickselect(std::vector<std::int64_t> data, std::size_t k);

/// @brief The median-of-medians (BFPRT) variant of @ref quickselect: identical k-semantics and
///        identical return value, but with a *guaranteed worst-case O(n)* running time. Blum,
///        Floyd, Pratt, Rivest & Tarjan (1973) removed Quickselect's quadratic worst case by
///        replacing the pivot heuristic with a provably good pivot: split the range into groups of
///        five, find each group's median (a constant-time sort of at most five elements), then
///        recursively take the *median of those medians* as the pivot. That pivot is guaranteed to
///        be greater than at least (roughly) 3n/10 elements and less than at least 3n/10, so every
///        partition discards a constant fraction of the array. The resulting recurrence
///        T(n) <= T(n/5) + T(7n/10) + O(n) solves to O(n) because 1/5 + 7/10 < 1. The constant is
///        larger than plain Quickselect's, so this variant trades average-case speed for an
///        adversary-proof worst case; both functions always agree on the answer.
///
/// @param data the values to select from (taken by value; reordered internally, caller's copy is
///        unaffected).
/// @param k the 0-indexed rank to retrieve (0 = minimum, data.size()-1 = maximum).
/// @return the k-th smallest element of @p data (equal to what @ref quickselect returns).
/// @throws std::invalid_argument if @p data is empty or if k >= data.size().
[[nodiscard]] std::int64_t quickselect_mom(std::vector<std::int64_t> data, std::size_t k);

} // namespace datamunge::algorithms
