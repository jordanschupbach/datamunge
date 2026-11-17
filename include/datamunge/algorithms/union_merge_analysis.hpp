#pragma once
/// @file union_merge_analysis.hpp
/// @brief Checked native union merge analysis and actual producing diagnostics.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/merge.hpp>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Generate the original sorted overlapping-list workload.
/// Next state=(1103515245*state+12345) mod2^31, then modulo range. This repairs undefined signed overflow in the original report; finite modulo sampling is biased. O(rows*length*log length), calls independent.
/// @param range Positive value range.
/// @param rows Number of runs, default5; <=10000.
/// @param length Keys per run, default100; total<=1000000.
/// @param seed 31-bit LCG initial state, default1.
/// @return Owned sorted rows; arithmetic uses explicit uint32 wrap then31-bit mask.
/// @throws std::invalid_argument Zero range.
/// @throws std::length_error Input budget exceeded.
/// @par Example
/// @code
/// auto lists=dm::algorithms::union_overlap_lists(20);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline std::vector<std::vector<int>> union_overlap_lists(
    int           range,
    std::size_t   rows   = 5,
    std::size_t   length = 100,
    std::uint32_t seed   = 1) {
  if (range <= 0)
    throw std::invalid_argument("overlap range");
  if (rows > 10000 || (rows && length > 1000000 / rows))
    throw std::length_error("overlap budget");
  std::vector<std::vector<int>> out(rows);
  for (auto& row : out) {
    for (std::size_t j = 0; j < length; ++j) {
      seed = (seed * UINT32_C(1103515245) + 12345) & UINT32_C(0x7fffffff);
      row.push_back(seed % range);
    }
    std::sort(row.begin(), row.end());
  }
  return out;
}

} // namespace datamunge::algorithms
