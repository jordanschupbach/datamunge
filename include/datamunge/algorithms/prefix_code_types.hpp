#pragma once
/// @file prefix_code_types.hpp
/// @brief Shared actual emitted-word boundaries for native prefix codecs.
#include <cstddef>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief One actual complete codeword emission in its producing bit stream.
/// @par Example
/// @code
/// const dm::algorithms::PrefixWordSpan value{};
/// std::print("{}\n",value.symbol); // 0
/// @endcode
struct PrefixWordSpan {
  std::size_t symbol{}; ///< Actual nonnegative alphabet index.
  std::size_t begin{};  ///< Inclusive logical bit offset.
  std::size_t end{};    ///< Exclusive actual word-end offset.
};
} // namespace datamunge::algorithms
