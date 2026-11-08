/// @file integer_arithmetic.cpp
/// @brief Division-based integer root search and overflow-safe decimal conversion.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/integer_arithmetic.hpp>
namespace datamunge::algorithms {
std::uint64_t integer_sqrt_u64(std::uint64_t n) noexcept {
  std::uint64_t low = 0, high = std::uint64_t{1} << 32;
  while (low + 1 < high) {
    const auto middle = low + (high - low) / 2;
    if (middle <= n / middle)
      low = middle;
    else
      high = middle;
  }
  return low;
}
std::uint64_t integer_sqrt_u128(unsigned __int128 n) noexcept {
  if (!n)
    return 0;
  const auto upper = static_cast<std::uint64_t>(n >> 64);
  const auto bits =
      upper ? 64 + std::bit_width(upper) : std::bit_width(static_cast<std::uint64_t>(n));
  auto high = static_cast<unsigned __int128>(1) << ((bits + 1) / 2);
  auto low  = high / 2;
  while (low + 1 < high) {
    const auto middle = low + (high - low) / 2;
    if (middle <= n / middle)
      low = middle;
    else
      high = middle;
  }
  return static_cast<std::uint64_t>(low);
}
std::string uint128_text(unsigned __int128 value) {
  std::string result;
  do {
    result.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  } while (value);
  std::reverse(result.begin(), result.end());
  return result;
}
std::string int128_text(__int128 value) {
  const auto magnitude = value < 0 ? static_cast<unsigned __int128>(-(value + 1)) + 1
                                   : static_cast<unsigned __int128>(value);
  auto       result    = uint128_text(magnitude);
  if (value < 0)
    result.insert(result.begin(), '-');
  return result;
}
} // namespace datamunge::algorithms
