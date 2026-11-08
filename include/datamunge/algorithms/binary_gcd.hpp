#pragma once
/// @file binary_gcd.hpp
/// @brief Stein's unsigned 64-bit greatest common divisor and its canonical execution trace.
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {
/// @brief One binary-GCD subtraction after removing twos and ordering the odd operands.
/// Values are exact unsigned integers. The next state is (smaller_odd,difference).
/// @par Example
/// @code
/// const auto trace = dm::algorithms::binary_gcd_trace(54,24);
/// std::print("{} {}\n", trace.steps.front().smaller_odd, trace.steps.front().difference); // 3 24
/// @endcode
/// @see BinaryGcdTrace, binary_gcd_trace()
struct BinaryGcdStep {
  std::uint64_t smaller_odd{0}; ///< Smaller odd operand after the optional swap.
  std::uint64_t larger_odd{0}; ///< Larger odd operand before subtraction.
  std::uint64_t difference{0}; ///< larger_odd-smaller_odd; zero terminates the loop.
  unsigned removed_twos{0}; ///< Trailing zero bits stripped from the incoming second operand before swapping.
};

/// @brief Owned result and chronological subtraction states from the canonical binary-GCD loop.
/// Zero-input cases return no steps, zero common_twos and zero initial_odd_a.
/// Otherwise gcd equals the final smaller_odd shifted left by common_twos.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::binary_gcd_trace(54,24);
/// std::print("{} {}\n", trace.gcd, trace.common_twos); // 6 1
/// @endcode
/// @see binary_gcd_trace()
struct BinaryGcdTrace {
  std::uint64_t gcd{0}; ///< Greatest common divisor, including gcd(0,0)=0.
  unsigned common_twos{0}; ///< Shared power-of-two exponent removed before the loop.
  std::uint64_t initial_odd_a{0}; ///< First operand after removing all its trailing zeros.
  std::vector<BinaryGcdStep> steps; ///< One entry per subtraction, including the terminal zero difference.
};

/// @brief Compute a greatest common divisor using Stein's shifts and subtractions.
/// Supports the entire uint64_t range without division, multiplication or overflow.
/// gcd(0,b)=b, gcd(a,0)=a, and gcd(0,0)=0. Count-trailing-zero operations are used
/// only on nonzero words. For w-bit operands, O(w) word operations and O(1)
/// auxiliary storage; this is not a multiprecision bit-complexity claim.
/// Inputs are values, no shared state is mutated, and no allocation is performed.
/// @param a First nonnegative integer, represented as uint64_t.
/// @param b Second nonnegative integer, represented as uint64_t.
/// @return Exact greatest common divisor; zero only when both inputs are zero.
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::binary_gcd(1071,462)); // 21
/// @endcode
/// @par References
/// - J. Stein (1967), Computational problems associated with Racah algebra,
///   Journal of Computational Physics 1(3), 397–405, doi:10.1016/0021-9991(67)90047-2.
/// @see binary_gcd_trace()
std::uint64_t binary_gcd(std::uint64_t a,std::uint64_t b);

/// @brief Record the same binary-GCD loop used by binary_gcd(), without report-side replay.
/// Trace states are exact; removed_twos refers to the second operand before the
/// optional swap. O(w) word operations and O(w) owned trace storage for w-bit
/// inputs. Zero-input cases return the direct answer with no subtraction states.
/// @param a First nonnegative uint64_t operand.
/// @param b Second nonnegative uint64_t operand.
/// @return Owned result, initial normalisation and every completed subtraction.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::binary_gcd_trace(1071,462);
/// std::print("{} {}\n", trace.gcd, trace.steps.size()); // 21 5
/// @endcode
/// @par References
/// - J. Stein (1967), Computational problems associated with Racah algebra,
///   Journal of Computational Physics 1(3), 397–405, doi:10.1016/0021-9991(67)90047-2.
/// @see binary_gcd(), BinaryGcdStep
BinaryGcdTrace binary_gcd_trace(std::uint64_t a,std::uint64_t b);
} // namespace datamunge::algorithms
