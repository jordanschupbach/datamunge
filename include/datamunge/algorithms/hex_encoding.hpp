#pragma once
/// @file hex_encoding.hpp
/// @brief Shared lowercase hexadecimal encoding for byte arrays and binary strings.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
namespace datamunge::algorithms {
/// @brief Encode a binary string/view as lowercase hexadecimal without losing NUL bytes.
/// Each unsigned byte becomes exactly two characters, high nibble first. Empty gives
/// empty; O(n) work/output storage, borrowed input unchanged, no shared state. This
/// presentation helper uses byte-indexed lookup; it is separate from cipher kernels.
/// @param bytes Borrowed binary view, valid during the call.
/// @return Owned lowercase hexadecimal string of length 2*bytes.size().
/// @throws std::length_error Encoded size exceeds std::string capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::to_hex(std::string_view("\0\x80\xff",3))); // 0080ff
/// @endcode
/// @see byte_histogram()
inline std::string to_hex(std::string_view bytes) {
  /// @brief Fixed lowercase symbols used only for output formatting.
  static constexpr char digits[]="0123456789abcdef";
  std::string output;if(bytes.size()>output.max_size()/2)throw std::length_error("to_hex: encoded size exceeds string capacity");
  output.reserve(bytes.size()*2);
  for(unsigned char byte:bytes) {output.push_back(digits[byte>>4]);output.push_back(digits[byte&15]);}
  return output;
}
/// @brief Encode a fixed-size byte digest with the original lowercase-hex array API.
/// Same encoding as the binary-view overload, including empty arrays. Owned output,
/// input unchanged, no shared state; O(N) work/output storage. The view is used only
/// during this call and is never retained.
/// @tparam N Number of unsigned bytes, possibly zero.
/// @param digest Borrowed fixed-size byte array.
/// @return Owned lowercase hexadecimal string of exactly 2*N characters.
/// @throws std::length_error Encoded size exceeds std::string capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::to_hex(std::array<std::uint8_t,3>{0,128,255})); // 0080ff
/// @endcode
/// @see to_hex(std::string_view)
template<std::size_t N>
inline std::string to_hex(const std::array<std::uint8_t,N>& digest) {
  if constexpr(N==0)return {};
  else return to_hex(std::string_view(reinterpret_cast<const char*>(digest.data()),N));
}
} // namespace datamunge::algorithms
