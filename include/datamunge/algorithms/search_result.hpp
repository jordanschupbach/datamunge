#pragma once
/// @file search_result.hpp
/// @brief Shared result contract for integer sequence searchers.
#include <cstddef>

namespace datamunge::algorithms {
/// @brief Owned outcome of an integer sequence search, with an element-probe count.
/// A successful index identifies a matching element in the searched representation;
/// an Eytzinger search reports its layout position, not the original sorted index.
/// An absent key has found=false and index=-1. Probes count inspected elements,
/// not CPU cycles or primitive comparisons; some searchers can inspect an element
/// more than once. The record owns no input data and retains no borrowed pointers.
/// @par Example
/// @code
/// const auto result = dm::algorithms::binary_search({1,3,5},3);
/// std::print("{} {} {}\n", result.found, result.index, result.probes); // true 1 1
/// @endcode
/// @see binary_search(), jump_search(), eytzinger_search()
struct SearchResult {
  bool found{false}; ///< Whether a matching element was located.
  std::ptrdiff_t index{-1}; ///< Matching index in the searched representation; -1 when absent.
  std::size_t probes{0}; ///< Number of element inspections recorded by the searcher.
};
} // namespace datamunge::algorithms
