#pragma once
/// @file modular_exponentiation.hpp
/// @brief Exact unsigned 64-bit modular products and actual right-to-left powering traces.
/// Uses unsigned __int128 products, as required by this build. Exponent-dependent
/// branches, counts and trace contents make these routines unsuitable as constant-time
/// secret-key primitives. No primality, group or protocol validation is performed.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual operation counts of right-to-left square-and-multiply.
/// Each processed bit performs one square, including the final unused square.
/// Set bits additionally multiply the accumulator. Counts exclude input reduction,
/// division internals, copying and allocation; these are not timings.
/// @par Example
/// @code
/// const auto t=dm::algorithms::mod_pow_trace(5,6,23);
/// std::print("{} {} {} {}\n",t.work.bits,t.work.squarings,t.work.accumulator_products,t.work.products); // 3 3 2 5
/// @endcode
/// @see mod_pow_trace()
struct ModPowWork {
  std::size_t bits{0}; ///< Actual loop iterations; zero for exponent zero or modulus one.
  std::size_t squarings{0}; ///< Actual running-base modular squares.
  std::size_t accumulator_products{0}; ///< Actual accumulator modular products on set bits.
  std::size_t products{0}; ///< Total squares plus accumulator products.
};
/// @brief One actual low-bit fold, running-base square and exponent shift.
/// Before the step, accumulator*running^remaining equals the original power modulo m.
/// multiply is the low bit of remaining_before; squared_base is computed even for
/// the final bit. No independent replay is used to obtain these values.
/// @par Example
/// @code
/// const auto t=dm::algorithms::mod_pow_trace(5,6,23);
/// std::print("{} {} {} {}\n",t.steps[1].remaining_before,t.steps[1].running_before,t.steps[1].accumulator_after,t.steps[1].squared_base); // 3 2 2 4
/// @endcode
/// @see ModPowTrace
struct ModPowStep {
  std::size_t bit_index{0}; ///< Zero-based bit index, least significant first.
  std::uint64_t remaining_before{0}; ///< Exponent suffix before shifting right.
  std::uint64_t running_before{0}; ///< Current base^(2^bit_index) residue.
  std::uint64_t accumulator_before{0}; ///< Accumulator before the optional multiply.
  bool multiply{false}; ///< True exactly when the current exponent bit is one.
  std::uint64_t accumulator_after{0}; ///< Actual accumulator after the optional multiply.
  std::uint64_t squared_base{0}; ///< Actual running-base square modulo m.
};
/// @brief Owned scalar inputs, actual exponent-bit states and final modular residue.
/// No replay or shared state. All trace data, including exponent bits, is exposed.
/// Modulus one has result/initial_base zero and no steps/products, regardless of
/// exponent; exponent zero otherwise returns one with no steps/products.
/// @par Example
/// @code
/// const auto t=dm::algorithms::mod_pow_trace(5,6,23);
/// std::print("{} {} {}\n",t.initial_base,t.result,t.steps.size()); // 5 8 3
/// @endcode
/// @see mod_pow(), mod_pow_trace()
struct ModPowTrace {
  std::uint64_t base{0}; ///< Original unreduced input base.
  std::uint64_t exponent{0}; ///< Original exponent, including zero.
  std::uint64_t modulus{0}; ///< Original nonzero modulus in a native result.
  std::uint64_t initial_base{0}; ///< base modulo modulus, or zero for modulus one.
  std::uint64_t result{0}; ///< Exact final residue in [0,modulus).
  ModPowWork work; ///< Actual modular-product counts.
  std::vector<ModPowStep> steps; ///< Actual least-significant-bit-first transitions.
};
/// @brief Compute (a*b) modulo a nonzero modulus without unsigned multiplication wrap.
/// Forms the complete product in unsigned __int128. Inputs may be unreduced and
/// span the full uint64 range; modulus one gives zero. O(1) word operations/storage
/// on this fixed-width implementation, without a constant-time guarantee.
/// @param a First unsigned factor.
/// @param b Second unsigned factor.
/// @param m Nonzero modulus, up to UINT64_MAX.
/// @return Exact residue in [0,m).
/// @throws std::invalid_argument If m is zero, before remainder evaluation.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::mul_mod(22,22,23)); // 1
/// @endcode
/// @see mod_pow(), mod_pow_trace()
std::uint64_t mul_mod(std::uint64_t a,std::uint64_t b,std::uint64_t m);
/// @brief Compute base^exponent modulo a nonzero modulus by right-to-left binary powering.
/// Exact for every uint64 input/modulus using full 128-bit products. Keeps the
/// existing loop order, including one final unused square. For m>1 and e>0 there
/// are bit_width(e) squares and popcount(e) accumulator products; exponent zero
/// gives one, modulus one gives zero (also for 0^0). O(log(e)+1) word operations
/// and O(1) driver storage. No constant-time or protocol-security claim.
/// @param base Arbitrary unsigned base; reduced once before positive-exponent work.
/// @param exponent Arbitrary unsigned exponent, including zero.
/// @param modulus Nonzero modulus, prime or composite.
/// @return Exact residue in [0,modulus).
/// @throws std::invalid_argument If modulus is zero.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::mod_pow(5,6,23),dm::algorithms::mod_pow(0,0,1)); // 8 0
/// @endcode
/// @par References
/// - W. Diffie and M. E. Hellman (1976), New Directions in Cryptography, IEEE IT22(6), 644–654, doi:10.1109/TIT.1976.1055638 (repeated-squaring exponentiation).
/// @see mod_pow_trace(), mul_mod()
std::uint64_t mod_pow(std::uint64_t base,std::uint64_t exponent,std::uint64_t modulus);
/// @brief Record actual right-to-left exponentiation states and operation counts.
/// Uses the same kernel as mod_pow(), without replay. Complete effective bit count
/// (at most64; zero for modulus one) is checked against the cap before allocation
/// or powering. Zero disables only the optional cap. O(log(e)+1) work/storage.
/// @param base Arbitrary unsigned base, retained unreduced in the result.
/// @param exponent Arbitrary unsigned exponent; all bits become observable in the trace.
/// @param modulus Nonzero prime or composite modulus.
/// @param max_steps Complete bit-record cap, default64; zero disables.
/// @return Owned actual bit transitions and exact result, including zero-work edge cases.
/// @throws std::invalid_argument If modulus is zero, before cap checks.
/// @throws std::length_error If the effective bit count exceeds a nonzero cap.
/// @throws std::bad_alloc If history allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::mod_pow_trace(19,6,23,3);
/// std::print("{} {}\n",t.result,t.work.products); // 2 5
/// @endcode
/// @par References
/// - W. Diffie and M. E. Hellman (1976), New Directions in Cryptography, doi:10.1109/TIT.1976.1055638.
/// @see mod_pow(), ModPowTrace
ModPowTrace mod_pow_trace(std::uint64_t base,std::uint64_t exponent,std::uint64_t modulus,std::size_t max_steps=64);
} // namespace datamunge::algorithms
