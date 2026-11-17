#pragma once
/// @file montgomery.hpp
/// @brief Reusable unsigned64-bit Montgomery contexts with carry-safe REDC traces.
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual carry-safe REDC state for a product of two reduced residues.
/// The exact sum T+q*n can require129 bits. sum_carry retains that bit;
/// quotient_low retains the low64 bits after division by2^64. A subtraction
/// occurs if the carry is set or quotient_low>=n. These are owned observations.
/// @par Example
/// @code
/// dm::algorithms::MontgomeryContext c(17);
/// auto t=c.reduce_trace(3,4);
/// std::print("{}\n",t.result); // 12
/// @endcode
/// @see MontgomeryContext::reduce_trace()
struct MontgomeryReduction {
  std::uint64_t left{0};           ///< Reduced first input.
  std::uint64_t right{0};          ///< Reduced second input.
  std::uint64_t product_low{0};    ///< Low64 product bits.
  std::uint64_t product_high{0};   ///< High64 product bits.
  std::uint64_t correction{0};     ///< q=product_low*nprime modulo2^64.
  bool          sum_carry{false};  ///< Actual129th bit of product+q*n.
  std::uint64_t quotient_low{0};   ///< Low64 quotient bits before correction subtraction.
  bool          subtracted{false}; ///< Whether the final modulus subtraction occurs.
  std::uint64_t result{0};         ///< Actual reduced product times R inverse modulo n.
};
/// @brief Actual powering step in the Montgomery domain.
/// @par Example
/// @code
/// dm::algorithms::MontgomeryContext c(17);
/// auto t=c.power_trace(3,2);
/// std::print("{}\n",t.steps.size()); // 2
/// @endcode
/// @see MontgomeryPowerTrace
struct MontgomeryPowerStep {
  std::uint64_t remaining{0};          ///< Actual exponent suffix before shift.
  std::uint64_t running{0};            ///< Actual running Montgomery residue before square.
  std::uint64_t accumulator_before{0}; ///< Actual accumulator before optional fold.
  bool          multiplied{false};     ///< True when low exponent bit is one.
  std::uint64_t accumulator_after{0};  ///< Actual accumulator after optional fold.
  std::uint64_t squared{0}; ///< Actual running residue after square, including final unused square.
};
/// @brief Owned actual exponentiation states with one shared modulus context.
/// One conversion into the domain and one conversion out surround the binary
/// loop. Counts are actual REDC calls, not CPU instructions or timings.
/// @par Example
/// @code
/// auto t=dm::algorithms::MontgomeryContext(3233).power_trace(123,17);
/// std::print("{} {}\n",t.result,t.reductions); // 855 9
/// @endcode
/// @see MontgomeryContext::power_trace()
struct MontgomeryPowerTrace {
  std::uint64_t base{0};                  ///< Original unreduced base.
  std::uint64_t exponent{0};              ///< Original exponent.
  std::uint64_t modulus{0};               ///< Context modulus.
  std::uint64_t initial_running{0};       ///< Base times R modulo modulus.
  std::uint64_t initial_accumulator{0};   ///< One times R modulo modulus.
  std::uint64_t result{0};                ///< Decoded ordinary residue.
  std::size_t   reductions{0};            ///< Actual REDC calls, including domain entry/exit.
  std::vector<MontgomeryPowerStep> steps; ///< Actual low-bit-first transitions.
};
/// @brief Precompute one odd modulus for carry-safe Montgomery arithmetic.
/// R=2^64. Setup computes nprime=-n inverse modulo R using five Newton
/// refinements from the odd seed n, plus R and R squared modulo n. Setup and
/// arbitrary-input reduction use remainder operations; only the reduced-domain
/// REDC kernel avoids division. All inputs/results use unsigned64-bit words,
/// with full128-bit products and an explicit129th sum bit. No constant-time
/// guarantee. Methods do not mutate the context and are independently callable.
/// @par Example
/// @code
/// dm::algorithms::MontgomeryContext c(3233);
/// std::print("{} {}\n",c.multiply(123,456),c.power(123,17)); // 1127 855
/// @endcode
/// @par References
/// - P. L. Montgomery (1985), Modular multiplication without trial division, Mathematics of Computation44,519–521, doi:10.1090/S0025-5718-1985-0777282-X.
/// @see montgomery_multiply()
class MontgomeryContext {
 public:
  /// @brief Validate and precompute an odd modulus, including modulus one.
  /// @param modulus Odd nonzero modulus, up to UINT64_MAX.
  /// @throws std::invalid_argument If modulus is zero or even.
  /// @par Example
  /// @code
  /// dm::algorithms::MontgomeryContext c(17);
  /// std::print("{}\n",c.modulus()); // 17
  /// @endcode
  /// @see MontgomeryContext
  explicit MontgomeryContext(std::uint64_t modulus);
  /// @brief Return the immutable odd modulus.
  /// @return Context modulus.
  std::uint64_t modulus() const noexcept { return n_; }
  /// @brief Return minus the modulus inverse modulo2^64.
  /// @return Precomputed correction multiplier.
  std::uint64_t nprime() const noexcept { return nprime_; }
  /// @brief Return R squared modulo the modulus.
  /// @return Setup residue used for domain entry.
  std::uint64_t r_squared() const noexcept { return r2_; }
  /// @brief Return the actual seed and five successive Newton inverse iterates.
  /// @return Owned fixed-size inverse history, ending at n inverse modulo R.
  std::array<std::uint64_t, 6> inverse_steps() const noexcept { return inverse_; }
  /// @brief Compute a*b*R inverse modulo n for already reduced residues.
  /// O(1) word work/storage; no division inside the REDC kernel.
  /// @param a Residue strictly below n, including0 when n=1.
  /// @param b Residue strictly below n.
  /// @return Reduced Montgomery product.
  /// @throws std::invalid_argument If either operand is not reduced.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::algorithms::MontgomeryContext(17).reduce(3,4)); // 12
  /// @endcode
  /// @see reduce_trace()
  std::uint64_t reduce(std::uint64_t a, std::uint64_t b) const;
  /// @brief Capture the actual REDC product, correction and129-bit carry.
  /// Uses the same kernel as reduce(), without replay; O(1) work/storage.
  /// @param a Reduced first residue, strictly below n.
  /// @param b Reduced second residue, strictly below n.
  /// @return Owned actual reduction state.
  /// @throws std::invalid_argument If either operand is not reduced.
  /// @par Example
  /// @code
  /// auto t=dm::algorithms::MontgomeryContext(17).reduce_trace(3,4);
  /// std::print("{}\n",t.result); // 12
  /// @endcode
  /// @see reduce(), MontgomeryReduction
  MontgomeryReduction reduce_trace(std::uint64_t a, std::uint64_t b) const;
  /// @brief Compute an ordinary modular product with two REDC calls.
  /// Inputs are first reduced using remainder; one factor enters the domain.
  /// O(1) fixed-word work/storage, including input remainder operations.
  /// @param a Arbitrary unsigned factor.
  /// @param b Arbitrary unsigned factor.
  /// @return Exact a*b modulo n; zero for modulus one.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::algorithms::MontgomeryContext(1000003).multiply(123,456)); // 56088
  /// @endcode
  /// @see power(), montgomery_multiply()
  std::uint64_t multiply(std::uint64_t a, std::uint64_t b) const;
  /// @brief Raise an arbitrary base using one domain entry and exit.
  /// Binary low-bit-first loop retains the final unused square. Modulus one
  /// returns zero, including exponent zero. O(log exponent+1) word work,
  /// O(1) driver storage; exponent-dependent branches are observable.
  /// @param base Arbitrary unsigned base.
  /// @param exponent Arbitrary unsigned exponent.
  /// @return Exact base^exponent modulo n.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::algorithms::MontgomeryContext(3233).power(855,2753)); // 123
  /// @endcode
  /// @see power_trace()
  std::uint64_t power(std::uint64_t base, std::uint64_t exponent) const;
  /// @brief Capture actual binary powering in the Montgomery domain.
  /// Same producer as power(), without replay. For n>1, REDC count is
  /// 2+bit_width(exponent)+popcount(exponent), including exponent zero.
  /// For n=1 all counts are zero. History uses O(log exponent+1) space.
  /// @param base Arbitrary base, retained unreduced.
  /// @param exponent Arbitrary exponent, retained in the result.
  /// @param max_steps Complete bit-record cap, default64; zero disables.
  /// @return Owned actual states, residue and REDC count.
  /// @throws std::length_error If effective bit width exceeds a nonzero cap, before powering.
  /// @throws std::bad_alloc If history allocation fails.
  /// @par Example
  /// @code
  /// auto t=dm::algorithms::MontgomeryContext(17).power_trace(3,2);
  /// std::print("{} {}\n",t.result,t.reductions); // 9 5
  /// @endcode
  /// @see MontgomeryPowerTrace, power()
  MontgomeryPowerTrace
  power_trace(std::uint64_t base, std::uint64_t exponent, std::size_t max_steps = 64) const;

 private:
  std::uint64_t                n_;       ///< Immutable validated modulus.
  std::uint64_t                nprime_;  ///< Negative modular inverse.
  std::uint64_t                r_;       ///< R modulo n.
  std::uint64_t                r2_;      ///< R squared modulo n.
  std::array<std::uint64_t, 6> inverse_; ///< Actual Newton states.

  /// @brief Shared optional-capture REDC producer for reduced operands.
  /// @param a Reduced first operand.
  /// @param b Reduced second operand.
  /// @param out Optional actual-state sink.
  /// @return Canonical residue.
  std::uint64_t redc(std::uint64_t a, std::uint64_t b, MontgomeryReduction* out) const;
  /// @brief Shared optional-capture binary-power producer.
  /// @param base Arbitrary base.
  /// @param exponent Arbitrary exponent.
  /// @param out Optional actual-state sink.
  /// @return Decoded canonical residue.
  std::uint64_t run_power(std::uint64_t base, std::uint64_t exponent, MontgomeryPowerTrace* out)
      const;
};
/// @brief Compute an ordinary modular product with a fresh Montgomery context.
/// Setup uses remainder operations on every call; use MontgomeryContext to
/// amortize setup across calls. Exact full-width inputs, O(1) word work/storage.
/// @param a Arbitrary unsigned first factor.
/// @param b Arbitrary unsigned second factor.
/// @param n Odd nonzero modulus, including one.
/// @return Exact ordinary a*b modulo n.
/// @throws std::invalid_argument If n is zero or even.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::montgomery_multiply(123,456,1000003)); // 56088
/// @endcode
/// @par References
/// - P. L. Montgomery (1985), doi:10.1090/S0025-5718-1985-0777282-X.
/// @see MontgomeryContext
std::uint64_t montgomery_multiply(std::uint64_t a, std::uint64_t b, std::uint64_t n);
} // namespace datamunge::algorithms
