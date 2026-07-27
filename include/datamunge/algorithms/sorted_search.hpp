#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief The outcome of searching a sorted sequence for a key. @ref found says whether the key is
///        present; @ref index is the position of a matching element (into the searched array) when
///        found, or -1 when absent; @ref probes counts how many array elements the search inspected
///        (compared against the key), a hardware-independent proxy for the algorithm's work that
///        lets the different searchers be compared directly.
struct SearchResult {
    bool           found{false}; ///< whether the key was located.
    std::ptrdiff_t index{-1};    ///< index of a matching element, or -1 if not found.
    std::size_t    probes{0};    ///< number of array elements examined during the search.
};

/// @brief Classic *binary search* on an ascending-sorted vector @p a: repeatedly halve the search
///        interval, comparing the key against the middle element. Runs in @c O(log n) comparisons
///        and @c O(1) extra space. @p a must be sorted in non-decreasing order; if the key occurs
///        more than once, the index of one of the occurrences is returned.
SearchResult binary_search(const std::vector<int>& a, int key);

/// @brief *Jump (block) search* on an ascending-sorted vector @p a: step forward in fixed blocks of
///        size @c floor(sqrt(n)) until a block whose last element is >= key is found, then scan that
///        block linearly. It makes @c O(sqrt(n)) comparisons -- worse than binary search but with a
///        purely forward access pattern, historically useful when jumping back is expensive (e.g.
///        tape or singly-linked storage).
SearchResult jump_search(const std::vector<int>& a, int key);

/// @brief *Interpolation (predictive) search* on an ascending-sorted vector @p a: instead of always
///        probing the middle, it estimates the key's position by linear interpolation between the
///        current endpoints' values, @c pos = lo + (key - a[lo]) * (hi - lo) / (a[hi] - a[lo]). On
///        (near-)uniformly distributed data this converges in @c O(log log n) expected comparisons;
///        in the worst case (very skewed data) it degrades to @c O(n). A division-by-zero guard
///        handles equal endpoints. @p a must be sorted in non-decreasing order.
SearchResult interpolation_search(const std::vector<int>& a, int key);

/// @brief *Fibonacci search* on an ascending-sorted vector @p a: a divide-and-conquer search that
///        narrows the interval using consecutive Fibonacci numbers instead of halving, so it only
///        ever moves the probe by Fibonacci offsets and never needs division -- historically an
///        advantage on hardware where division was costly. It uses @c O(log n) comparisons.
///        @p a must be sorted in non-decreasing order.
SearchResult fibonacci_search(const std::vector<int>& a, int key);

/// @brief Build the *Eytzinger (BFS/heap) layout* of an ascending-sorted vector @p sorted: the array
///        of a complete binary search tree written in breadth-first order, so that node @c k (using
///        1-based indices) has children @c 2k and @c 2k+1. Filling the layout by an in-order
///        traversal places the sorted values at the tree nodes. This layout makes
///        @ref eytzinger_search cache-friendly: the children of a node are adjacent in memory and a
///        single cache line holds several successively-visited nodes.
///
/// @param sorted an ascending-sorted vector.
/// @return the Eytzinger-ordered array (same length and multiset of values as @p sorted).
std::vector<int> eytzinger_layout(const std::vector<int>& sorted);

/// @brief *Eytzinger binary search* over an array @p layout produced by @ref eytzinger_layout:
///        starting at the root (index 0 == node 1), move to the left child @c 2k when the key is
///        smaller and the right child @c 2k+1 otherwise, following the implicit tree without any
///        arithmetic on interval midpoints. It performs @c O(log n) comparisons like binary search
///        but with a far more cache-friendly access pattern on large arrays. The returned @ref
///        SearchResult::index, when found, is the position within @p layout (not the original sorted
///        array).
SearchResult eytzinger_search(const std::vector<int>& layout, int key);

} // namespace datamunge::algorithms
