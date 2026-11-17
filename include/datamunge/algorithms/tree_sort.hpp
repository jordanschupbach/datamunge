#pragma once
/// @file tree_sort.hpp
/// @brief Checked tree sort operations and producer diagnostics.
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned unbalanced BST and its actual iterative in-order traversal.
/// Duplicates branch right. Node IDs follow insertion order; -1 marks no child. Inputs unchanged; calls independent. O(n^2) worst-case insertion and O(n) storage.
/// @par Example
/// @code
/// auto r=dm::algorithms::tree_sort_work({2,1}); // sorted {1,2}
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
struct TreeSortWork {
  std::vector<int>         values;         ///< Node keys in actual insertion order.
  std::vector<int>         left;           ///< Actual left-child IDs, -1 for absent.
  std::vector<int>         right;          ///< Actual right-child IDs, -1 for absent.
  std::vector<std::size_t> depth;          ///< Root depth0.
  std::vector<std::size_t> inorder;        ///< Actual emitted node IDs.
  std::vector<int>         sorted;         ///< Owned sorted keys.
  std::size_t              comparisons{0}; ///< Actual key comparisons.
};
/// @brief Insert integers into an unbalanced BST, then traverse it.
/// @param input Borrowed keys; empty accepted.
/// @param max_comparisons Positive work ceiling; default20000000.
/// @return Actual nodes, order and comparison count.
/// @throws std::length_error More than1000000 keys or work budget exhausted.
/// @par Example
/// @code
/// auto r=dm::algorithms::tree_sort_work({2,1}); // one comparison
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
TreeSortWork tree_sort_work(const std::vector<int>& input, std::size_t max_comparisons = 20000000);
/// @brief Return the same BST producer sorted keys.
/// @param a Owned integer keys; empty accepted.
/// @return Sorted owned keys.
/// @throws std::length_error Default producer budget exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tree_sort({2,1});
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
std::vector<int> tree_sort(std::vector<int> a);
} // namespace datamunge::algorithms
