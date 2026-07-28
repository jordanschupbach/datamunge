#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The *simple (two-way) merge*: given two already-sorted sequences, interleave them into a
///        single sorted sequence by repeatedly taking the smaller of the two front elements. It runs
///        in @c O(|a|+|b|) time and one linear pass, is *stable* (equal elements keep @p a before
///        @p b), and is the merge step at the heart of merge sort.
///
/// @param a,b the two input sequences, each assumed sorted ascending.
/// @return the merged sorted sequence containing every element of @p a and @p b.
std::vector<int> simple_merge(const std::vector<int>& a, const std::vector<int>& b);

/// @brief The *k-way merge*: merges @p lists (each already sorted ascending) into one sorted sequence
///        using a *min-heap* of the current front element of every list. Each of the @c n total
///        elements is pushed and popped once, and every heap operation costs @c O(log k), so the
///        whole merge is @c O(n log k) -- far better than the @c O(n·k) of repeatedly scanning all
///        fronts, and the standard way to merge sorted runs in external sorting.
///
/// @param lists the input sequences, each assumed sorted ascending (empty lists are ignored).
/// @return the merged sorted sequence containing every element of every list.
std::vector<int> k_way_merge(const std::vector<std::vector<int>>& lists);

/// @brief The *union merge*: like @ref k_way_merge, but *duplicates are dropped* -- the output is the
///        sorted *set union* of every input list, each distinct value appearing exactly once. It uses
///        the same @c O(n log k) min-heap, simply skipping any value equal to the one just emitted, so
///        it deduplicates *during* the merge without a second pass. This is the merge used when
///        combining sorted posting lists or index runs that must not repeat an element.
///
/// @param lists the input sequences, each assumed sorted ascending (empty lists are ignored).
/// @return the sorted union: every distinct value across all lists, once each.
std::vector<int> union_merge(const std::vector<std::vector<int>>& lists);

} // namespace datamunge::algorithms
