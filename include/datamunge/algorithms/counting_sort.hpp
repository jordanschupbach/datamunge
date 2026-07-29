#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Largest key range (k = max - min + 1) counting_sort will allocate a count array for.
///        The whole point of counting sort is a k-sized bucket array, so an astronomically large
///        range would trigger an enormous allocation for no benefit; past this cap the routine
///        throws instead. Counting sort is only practical when k is comparable to n.
inline constexpr std::int64_t kCountingSortMaxRange = 1'000'000'000; // one billion buckets

/// @brief Counting sort: a *stable*, non-comparison sort for integer keys in a bounded range.
///        Instead of comparing elements it tallies how often each key value occurs, prefix-sums
///        those tallies into the final starting position of each key's block, then places every
///        element directly. Running time and space are O(n + k), where n is the number of
///        elements and k = max - min + 1 is the width of the key range -- linear when k = O(n),
///        but wasteful (and, past @ref kCountingSortMaxRange, rejected) when the range dwarfs the
///        element count. It is the stable inner loop of radix sort.
///
///        This overload auto-detects the range by scanning for the minimum and maximum key.
/// @param data the integers to sort ascending, in place. Sorting is stable: equal keys keep
///        their original relative order (observable when the keys carry a payload, e.g. inside a
///        radix sort). An empty vector is left untouched.
void counting_sort(std::vector<std::int64_t>& data);

/// @brief Counting sort with a caller-supplied key range -- the form used as a radix-sort
///        building block, where the digit range is known up front. Behaves exactly like the
///        auto-detecting overload but skips the min/max scan.
/// @param data the integers to sort ascending, in place (stable). Empty input is a no-op.
/// @param min_value the smallest key that may appear (inclusive).
/// @param max_value the largest key that may appear (inclusive).
/// @throws std::invalid_argument if @p max_value < @p min_value, or if any element of @p data
///         lies outside [@p min_value, @p max_value].
/// @throws std::length_error if the range max - min + 1 exceeds @ref kCountingSortMaxRange.
void counting_sort(std::vector<std::int64_t>& data, std::int64_t min_value, std::int64_t max_value);

} // namespace datamunge::algorithms
