#include <datamunge/algorithms/tree_sort.hpp>
#pragma once
#include <datamunge/algorithms/bucket_sort.hpp>
#include <datamunge/algorithms/cycle_sort.hpp>
#include <datamunge/algorithms/pigeonhole_sort.hpp>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Tree sort* (binary-tree sort): inserts every element into a *binary search tree*, then
///        produces the sorted output by an in-order traversal (which visits a BST's keys in
///        ascending order). With a balanced tree it is @c O(n log n); with the plain unbalanced BST
///        used here it is @c O(n log n) on average but @c O(n^2) on already-sorted input (the tree
///        degenerates to a list). Duplicates are kept (inserted into the right subtree), making it
///        stable-by-value.
///
/// @param a the input values.
/// @return a sorted (non-decreasing) copy of @p a.

} // namespace datamunge::algorithms
