#pragma once
/// @file sfc64_step.hpp
/// @brief Owned actual word/counter transition of the native SFC64 engine.
#include <array>
#include <cstdint>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::random {
/// @brief One actual raw engine transition; excludes inherited distribution cache.
/// @par Example
/// @code
/// const dm::random::Sfc64Step value{};
/// std::print("{}\n",value.value); // 0
/// @endcode
struct Sfc64Step {
  std::array<std::uint64_t, 4> before{}; ///< Actual a,b,c,counter before mutation.
  std::uint64_t                value{};  ///< Actual a+b+counter modulo2^64.
  std::array<std::uint64_t, 4>
      after{}; ///< Actual xor-shift, ninefold, rotate-add and incremented counter words.
};
} // namespace datamunge::random
