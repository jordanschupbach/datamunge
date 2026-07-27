#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief *Bucket sort*: a distribution sort that scatters the input into a number of ordered
///        *buckets* by value range, sorts each bucket (here by insertion sort), and concatenates
///        them. When the values are roughly uniformly distributed over their range it runs in
///        expected @c O(n) time -- the buckets stay small -- degrading to @c O(n^2) only if
///        everything lands in one bucket. It is not in-place (it needs the buckets) and, as
///        implemented, stable within a bucket. This integer version uses @c n buckets spanning
///        @c [min, max].
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> bucket_sort(std::vector<int> a);

/// @brief *Pigeonhole sort*: a non-comparison sort for integer keys in a small range. It allocates
///        one "hole" per distinct value in @c [min, max], drops each element into its hole, then
///        reads the holes back in order. It runs in @c O(n + range) time and space -- exact, stable,
///        and very fast when the range is comparable to @c n, but wasteful when the range dwarfs the
///        element count. Unlike counting sort it moves the actual elements (which matters when they
///        carry satellite data).
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> pigeonhole_sort(std::vector<int> a);

/// @brief *Cycle sort*: an in-place sort that performs the *theoretically minimum number of writes*.
///        It decomposes the permutation-to-sorted into cycles: for each position it counts how many
///        elements are smaller (that is the element's final index), then rotates it into place,
///        displacing whatever was there, and continues around the cycle until it closes. Every
///        element is written to memory exactly once (unless already in place), which is decisive when
///        writes are far costlier than reads (EEPROM/flash wear). It always makes @c O(n^2)
///        comparisons.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> cycle_sort(std::vector<int> a);

/// @brief *Tree sort* (binary-tree sort): inserts every element into a *binary search tree*, then
///        produces the sorted output by an in-order traversal (which visits a BST's keys in
///        ascending order). With a balanced tree it is @c O(n log n); with the plain unbalanced BST
///        used here it is @c O(n log n) on average but @c O(n^2) on already-sorted input (the tree
///        degenerates to a list). Duplicates are kept (inserted into the right subtree), making it
///        stable-by-value.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.
std::vector<int> tree_sort(std::vector<int> a);

} // namespace datamunge::algorithms
