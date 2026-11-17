#pragma once
/// @file truncated_binary.hpp
/// @brief Checked truncated binary operations and producer diagnostics.
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Append one checked truncated-binary codeword.
/// k=floor(log2 n); u=2^(k+1)-n computed wide. n1 uses zero bits. No shared state. O(log n) time.
/// @param v Symbol below n.
/// @param n Positive alphabet size, entire uint64 range supported.
/// @param out Caller-owned bits; unchanged on invalid input.
/// @throws std::invalid_argument n0 or v>=n.
/// @par Example
/// @code
/// std::vector<bool> bits; dm::algorithms::truncated_binary_encode_value(4,5,bits); //111
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline void
truncated_binary_encode_value(std::uint64_t v, std::uint64_t n, std::vector<bool>& out) {
  if (!n || v >= n)
    throw std::invalid_argument("truncated binary: invalid symbol/alphabet");
  int  k     = std::bit_width(n) - 1;
  auto u     = static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << (k + 1)) - n);
  auto x     = v < u ? v : v + u;
  int  width = v < u ? k : k + 1;
  for (int i = width - 1; i >= 0; --i)
    out.push_back((x >> i) & 1);
}
/// @brief Encode an owned bit stream for known alphabet size.
/// @param values Symbols, all below n.
/// @param n Positive alphabet size.
/// @return Concatenated bits, empty accepted.
/// @throws std::invalid_argument Invalid alphabet or symbol.
/// @par Example
/// @code
/// auto bits=dm::algorithms::truncated_binary_encode({0,4},5);
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline std::vector<bool> truncated_binary_encode(
    const std::vector<std::uint64_t>& values,
    std::uint64_t                     n) {
  if (!n)
    throw std::invalid_argument("zero alphabet");
  std::vector<bool> out;
  for (auto v : values)
    truncated_binary_encode_value(v, n, out);
  return out;
}
/// @brief Decode exactly count symbols from a complete bit stream.
/// O(count*log n) time and O(count) output. No partial result on error; input unchanged, calls independent.
/// @param bits Complete stream; trailing bits are rejected.
/// @param n Positive alphabet size; n1 decodes count zeros from empty bits.
/// @param count Nonnegative output count, at most1000000.
/// @return Exactly count decoded symbols.
/// @throws std::invalid_argument Zero alphabet, negative count, truncated stream or trailing bits.
/// @throws std::length_error Output count exceeds1000000.
/// @par Example
/// @code
/// auto v=dm::algorithms::truncated_binary_decode({false,false},5,1); //{0}
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline std::vector<std::uint64_t>
truncated_binary_decode(const std::vector<bool>& bits, std::uint64_t n, int count) {
  if (!n || count < 0)
    throw std::invalid_argument("invalid alphabet/count");
  if (count > 1000000)
    throw std::length_error("decode count limit");
  int         k    = std::bit_width(n) - 1;
  auto        u    = static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << (k + 1)) - n);
  std::size_t pos  = 0;
  auto        read = [&]() {
    if (pos == bits.size())
      throw std::invalid_argument("truncated stream");
    return static_cast<std::uint64_t>(bits[pos++]);
  };
  std::vector<std::uint64_t> out;
  for (int c = 0; c < count; ++c) {
    std::uint64_t x = 0;
    for (int j = 0; j < k; ++j)
      x = (x << 1) | read();
    if (x >= u)
      x = ((x << 1) | read()) - u;
    out.push_back(x);
  }
  if (pos != bits.size())
    throw std::invalid_argument("trailing bits");
  return out;
}

} // namespace datamunge::algorithms
