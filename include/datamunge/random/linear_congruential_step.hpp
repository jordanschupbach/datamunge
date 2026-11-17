#pragma once
/// @file linear_congruential_step.hpp
/// @brief Owned actual unsigned64 affine multiply/add operands and wrapped result.
#include <cstdint>
namespace datamunge::random {
/// @brief Actual low64 multiply/add transition from one LCG recurrence execution.
/// product and after use defined unsigned wrap modulo2^64. Wrap flags describe
/// exact unsigned range overflow, not a CPU flag/instruction or quality metric.
/// @par Example
/// @code
/// dm::random::LinearCongruential64 g(1,5,3);
/// const auto s=g.next_step();
/// std::print("{} {} {}\n",s.before,s.product,s.after); // 1 5 8
/// @endcode
/// @see LinearCongruential64::next_step(), lcg64_step()
struct LCG64Step {
  std::uint64_t before{0};     ///< Actual input state before the update.
  std::uint64_t multiplier{0}; ///< Actual caller/engine multiplier.
  std::uint64_t increment{0};  ///< Actual caller/engine increment.
  std::uint64_t product{0};    ///< Actual low64 product multiplier*before.
  std::uint64_t after{0};      ///< Actual low64 product+increment, committed/returned by engine.
  bool multiplication_wrapped{false}; ///< Whether exact nonnegative product exceeds UINT64_MAX.
  bool addition_wrapped{false}; ///< Whether exact product-low64 plus increment exceeds UINT64_MAX.
};
/// @brief Evaluate one pure unsigned64 affine transition, including explicit wrap evidence.
/// Unlike the full-period engine constructor, this diagnostic accepts arbitrary
/// unsigned coefficients, including0/even values for explicit failure controls.
/// No state mutation, allocation, borrowed output or shared state.
/// @param state Actual original unsigned64 state.
/// @param multiplier Actual arbitrary unsigned64 affine multiplier.
/// @param increment Actual arbitrary unsigned64 affine increment.
/// @return Owned actual operands, low64 product/output and exact overflow predicates.
/// @par Example
/// @code
/// const auto s=dm::random::lcg64_step(1,5,3);
/// std::print("{}\n",s.after); // 8
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, SIAM Review4(3),230–254, https://doi.org/10.1137/1004061.
/// @see LinearCongruential64::next_step()
LCG64Step
lcg64_step(std::uint64_t state, std::uint64_t multiplier, std::uint64_t increment) noexcept;
} // namespace datamunge::random
