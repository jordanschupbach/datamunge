#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Sorts @p data ascending, in place, using Quicksort (Hoare 1961) -- the classic
///        divide-and-conquer sort. A *pivot* is chosen and the range is *partitioned* so that every
///        element strictly less than the pivot precedes it and everything else follows; the pivot
///        thereby lands in its final sorted position and the two sides are sorted recursively. This
///        implementation makes three refinements that turn textbook Quicksort into a robust one.
///        First, *median-of-three* pivoting: the pivot is the median of the first, middle, and last
///        elements of the range, which makes the pathological O(n^2) behaviour on already-sorted or
///        reverse-sorted input astronomically unlikely while remaining fully deterministic (no
///        randomness, so results are reproducible). Second, *tail-call elimination*: after each
///        partition the smaller side is sorted by recursion and the larger side by looping, which
///        bounds the recursion depth to O(log n) and so prevents stack overflow on adversarial
///        inputs. Third, a *small-subarray cutoff*: ranges of at most sixteen elements are finished
///        with insertion sort, which is faster than recursion on tiny inputs. The average-case cost
///        is O(n log n) comparisons; the worst case is O(n^2) but effectively unreachable here.
///        Quicksort is *in place* (O(log n) auxiliary stack, no heap allocation) but *not stable*:
///        equal elements may be reordered relative to one another. Empty and single-element inputs
///        are returned unchanged.
///
/// @param data the array to sort in place (ascending).
void quicksort(std::vector<std::int64_t>& data);

/// @brief Identical to @ref quicksort(std::vector<std::int64_t>&) but additionally reports, in
///        @p comparisons, the total number of element-to-element comparisons performed (including
///        those made while selecting median-of-three pivots and during the insertion-sort cutoff).
///        This instrumentation exposes the algorithm's O(n log n) average behaviour for study and
///        plotting; the sort itself is unchanged. @p comparisons is overwritten (not accumulated).
///
/// @param data the array to sort in place (ascending).
/// @param comparisons out-parameter receiving the count of element comparisons performed.
void quicksort(std::vector<std::int64_t>& data, std::uint64_t& comparisons);

} // namespace datamunge::algorithms
