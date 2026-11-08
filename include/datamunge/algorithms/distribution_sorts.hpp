#pragma once

#include <vector>
#include <datamunge/algorithms/cycle_sort.hpp>
#include <datamunge/algorithms/bucket_sort.hpp>

namespace datamunge::algorithms {

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
