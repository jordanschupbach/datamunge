#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Rearranges @p data into a binary *max-heap* in place, in O(n) time. A max-heap is the
///        implicit array representation of a complete binary tree in which every parent is at
///        least as large as its children: the children of index i live at 2i+1 and 2i+2, so the
///        heap property is data[i] >= data[2i+1] and data[i] >= data[2i+2] whenever those
///        children exist. Bottom-up (Floyd's) construction sifts down every internal node from
///        the last one (index n/2 - 1) up to the root; because a node at height h costs O(h) and
///        most nodes are shallow, the total is the convergent sum of heights, giving O(n) rather
///        than O(n log n). Exposed separately from @ref heapsort so callers (and tests) can build
///        or inspect a heap directly. Empty and single-element inputs are already heaps.
///
/// @param data the array to heapify; modified in place.
void build_max_heap(std::vector<std::int64_t>& data);

/// @brief Checks whether @p data satisfies the binary max-heap property (data[i] >= its children
///        for every internal node). An empty or single-element array is trivially a heap.
///
/// @param data the array to test.
/// @return true iff @p data is a valid max-heap.
[[nodiscard]] bool is_max_heap(const std::vector<std::int64_t>& data);

/// @brief Sorts @p data ascending, in place, using Heapsort (Williams 1964; Floyd's linear-time
///        heap construction). The array is first turned into a max-heap in O(n) via
///        @ref build_max_heap; then the algorithm repeatedly swaps the root -- always the current
///        maximum -- with the last element of the active heap, shrinks the heap by one, and sifts
///        the new root down to restore the heap property. After n-1 such extractions the array is
///        sorted. Each of the n extractions costs O(log n) for the sift-down, so the algorithm
///        runs in O(n log n) in the *worst* case (unlike quicksort's O(n^2)) while using only
///        O(1) extra space (unlike merge sort's O(n)). The trade-offs are that Heapsort is *not*
///        stable and, because it jumps across the array following parent/child links, it is less
///        cache-friendly than quicksort in practice. Empty and single-element inputs are returned
///        unchanged.
///
/// @param data the array to sort in place (ascending).
void heapsort(std::vector<std::int64_t>& data);

} // namespace datamunge::algorithms
