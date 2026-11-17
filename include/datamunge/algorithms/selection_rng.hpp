#pragma once
/// @file selection_rng.hpp
/// @brief Shared historical xorshift selection stream, not a cryptographic generator.
#include <cstddef>
#include <cstdint>
namespace datamunge::algorithms::detail {
/// @brief Invocation-owned historical selection engine with a nonzero fallback seed.
/// @par Example
/// @code
/// dm::algorithms::detail::SelRng r{1};
/// (void)r;
/// @endcode
struct SelRng {
  std::uint64_t s; ///< Actual mutable xorshift state.

  /// @brief Initialize exact historical state; zero uses the fixed nonzero fallback.
  /// @param seed Complete unsigned seed.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::SelRng r{0};
  /// (void)r;
  /// @endcode
  explicit SelRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  /// @brief Advance the actual13/7/17 xorshift transition.
  /// @return Actual next64-bit word.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::SelRng r{1};
  /// (void)r.next();
  /// @endcode
  std::uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  /// @brief Draw one actual top53-bit half-open unit fraction.
  /// @return Actual double in[0,1), not a continuous-uniform mathematical draw.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::SelRng r{1};
  /// (void)r.uniform();
  /// @endcode
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  /// @brief Preserve the historical modulo-index primitive.
  /// @param n Strictly positive modulus, required by callers.
  /// @return Actual next word modulo n; modulo bias is not removed.
  /// @par Example
  /// @code
  /// dm::algorithms::detail::SelRng r{1};
  /// (void)r.below(3);
  /// @endcode
  std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};
} // namespace datamunge::algorithms::detail
