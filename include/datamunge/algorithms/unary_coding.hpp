#pragma once
/// @file unary_coding.hpp
/// @brief Checked native unary coding and actual producing diagnostics.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Append a complete unary word after checking its output budget.
/// Writes n ones then zero. O(n) time, no shared state.
/// @param n Number of leading ones.
/// @param out Caller bits; unchanged if word would exceed1000000 total bits.
/// @throws std::length_error Complete output would exceed1000000 bits.
/// @par Example
/// @code
/// std::vector<bool>b;dm::algorithms::unary_encode_value(3,b); //1110
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline void unary_encode_value(std::uint64_t n, std::vector<bool>& out) {
  if (out.size() >= 1000000 || n >= 1000000 - out.size())
    throw std::length_error("unary bit limit");
  out.insert(out.end(), static_cast<std::size_t>(n), true);
  out.push_back(false);
}
/// @brief Encode unary words into one owned complete stream.
/// @param values Nonnegative integers; empty accepted.
/// @return Concatenated unary words.
/// @throws std::length_error Complete stream exceeds1000000 bits.
/// @par Example
/// @code
/// auto b=dm::algorithms::unary_encode({0,1,3});
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline std::vector<bool> unary_encode(const std::vector<std::uint64_t>& values) {
  std::vector<bool> out;
  for (auto n : values)
    unary_encode_value(n, out);
  return out;
}
/// @brief Decode exactly count complete unary words.
/// O(bits+count) time; truncated words and trailing bits are errors, never partial success. Borrowed input unchanged; calls independent.
/// @param bits Complete stream with no trailing unread bits, at most1000000 bits.
/// @param count Nonnegative exact output count, at most1000000.
/// @return Exactly count decoded integers; empty only for count0.
/// @throws std::invalid_argument Negative count, missing terminator, too few words or trailing bits.
/// @throws std::length_error Input/output cap exceeded.
/// @par Example
/// @code
/// auto v=dm::algorithms::unary_decode({true,true,false},1); //{2}
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline std::vector<std::uint64_t> unary_decode(const std::vector<bool>& bits, int count) {
  if (count < 0)
    throw std::invalid_argument("negative unary count");
  if (count > 1000000 || bits.size() > 1000000)
    throw std::length_error("unary limit");
  std::vector<std::uint64_t> out;
  std::size_t                pos = 0;
  for (int c = 0; c < count; ++c) {
    std::uint64_t n = 0;
    while (pos < bits.size() && bits[pos]) {
      ++pos;
      ++n;
    }
    if (pos == bits.size())
      throw std::invalid_argument("missing unary delimiter");
    ++pos;
    out.push_back(n);
  }
  if (pos != bits.size())
    throw std::invalid_argument("trailing unary bits");
  return out;
}

} // namespace datamunge::algorithms
