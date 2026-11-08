#pragma once
/// @file byte_histogram.hpp
/// @brief Checked counts of binary byte values in equal-width groups.
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace datamunge::algorithms {
/// @brief Count binary byte values in equal-width contiguous groups over [0,255].
/// Group width is 256/bins; embedded NULs and signed-char high bits are counted as
/// unsigned bytes. Empty input gives bins zero counts after validating bins. Sum of
/// counts is exactly bytes.size(); no statistical test or uniformity inference is
/// performed. O(n+bins) work and O(bins) owned output, input unchanged, no shared state.
/// @param bytes Borrowed binary view, valid for the duration of the call.
/// @param bins Positive divisor of 256; default 256 counts each distinct byte value.
/// @return Owned counts in increasing byte-group order, exactly bins entries.
/// @throws std::invalid_argument bins is zero, above 256, or does not divide 256.
/// @par Example
/// @code
/// const auto h=dm::algorithms::byte_histogram(std::string_view("\0\x10\xff",3),16);
/// std::print("{} {} {}\n",h[0],h[1],h[15]); // 1 1 1
/// @endcode
/// @see to_hex()
inline std::vector<std::size_t> byte_histogram(std::string_view bytes,std::size_t bins=256) {
  if(!bins || bins>256 || 256%bins)throw std::invalid_argument("byte_histogram: bins must divide 256");
  std::vector<std::size_t> counts(bins,0);const auto width=256/bins;
  for(unsigned char byte:bytes)++counts[byte/width];
  return counts;
}
} // namespace datamunge::algorithms
