#pragma once
/// @file lagged_fibonacci_step.hpp
/// @brief Owned actual additive lagged-Fibonacci ring transition operands and positions.
#include <cstddef>
#include <cstdint>
namespace datamunge::random {
/// @brief Owned operands/output from one actual modular64 ring update.
/// value=long_word+short_word modulo2^64, overwriting write_index; short_index
/// identifies the other actual ring operand. No replay or engine references.
/// @par Example
/// @code
/// dm::random::LaggedFibonacci64 g(2026,2,5);const auto s=g.next_step();
/// std::print("{} {}\n",s.write_index,s.short_index); // 0 3
/// @endcode
/// @see LaggedFibonacci64::next_step()
struct LaggedFibonacciStep {
  std::size_t   write_index{0}; ///< Actual physical oldest-word slot overwritten.
  std::size_t   short_index{0}; ///< Actual physical slot holding the short-lag operand.
  std::uint64_t long_word{0};   ///< Actual original oldest ring word before overwrite.
  std::uint64_t short_word{0};  ///< Actual short-lag ring operand before overwrite.
  std::uint64_t value{0};       ///< Actual modular2^64 sum, also returned raw output.
  std::size_t   next_index{0};  ///< Actual next physical overwrite slot.
};
} // namespace datamunge::random
