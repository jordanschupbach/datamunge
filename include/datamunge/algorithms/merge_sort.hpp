#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Top-down (recursive) merge sort of @p data into ascending order, in place and *stable*.
///        The classic divide-and-conquer sort (von Neumann, 1945): split the range in half, sort
///        each half recursively, then *merge* the two sorted halves in linear time by repeatedly
///        taking the smaller front element. The merge draws from the left half first whenever the
///        two fronts are equal, so equal elements never change relative order -- the sort is
///        stable. A single O(n) scratch buffer is reused across all merges. Running time is
///        \f$\Theta(n\log n)\f$ on *every* input (best = average = worst), unlike quicksort whose
///        worst case is \f$\Theta(n^2)\f$; the price is \f$O(n)\f$ auxiliary space. Empty and
///        singleton ranges are already sorted and returned unchanged.
///
/// @param data the sequence to sort ascending in place.
void merge_sort(std::vector<std::int64_t>& data);

/// @brief Stable merge sort of keyed records by =first= (ascending), preserving the input order of
///        records that share a key. Provided mainly to *exhibit* the stability of @ref merge_sort:
///        set each record's =second= to its original index and, after sorting, records with equal
///        =first= will still appear in ascending =second= order. Same algorithm, buffer, and
///        tie-breaking rule as @ref merge_sort; only the comparison key differs.
///
/// @param data the records to sort in place by their =first= field.
void merge_sort_pairs(std::vector<std::pair<int, int>>& data);

} // namespace datamunge::algorithms
