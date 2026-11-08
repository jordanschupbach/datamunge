#pragma once
/// @file booth_multiplication.hpp
/// @brief Exact int32 Booth recoding, operation counts and canonical accumulator traces.
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Exact widened signed product and radix-2 recoding operation counts.
/// Counts depend on multiplier bit transitions, even when the multiplicand is
/// zero. An addition means signed digit +1, not necessarily a positive numeric
/// contribution when the multiplicand is negative. No inputs or shared state are modified.
/// @par Example
/// @code
/// const auto result=dm::algorithms::booth_multiply(-7,6);
/// std::print("{} {} {}\n",result.product,result.additions,result.subtractions); // -42 1 1
/// @endcode
/// @see booth_multiply(), BoothTrace
struct BoothResult {
  std::int64_t product{0}; ///< Exact signed int32-by-int32 product, always representable in int64.
  int additions{0}; ///< Number of (current,previous)=(0,1) pairs, giving digit +1.
  int subtractions{0}; ///< Number of (current,previous)=(1,0) pairs, giving digit -1.
};
/// @brief One actual multiplier-bit decision and its widened accumulator state.
/// previous is zero below bit zero, then the preceding lower bit. digit equals
/// previous-current. All 32 decisions are recorded, including zero contributions.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::booth_multiply_trace(3,7);
/// std::print("{} {}\n",trace.steps[0].digit,trace.steps[0].contribution); // -1 -3
/// @endcode
/// @see BoothTrace
struct BoothStep {
  unsigned bit_index{0}; ///< Bit position 0 through 31, in increasing execution order.
  unsigned current{0}; ///< Current bit of the uint32 representation of the multiplier.
  unsigned previous{0}; ///< Previous lower bit, with the initial implicit zero.
  int digit{0}; ///< Signed recoding digit: -1, 0 or +1.
  std::int64_t scaled_multiplicand{0}; ///< Multiplicand times 2^bit_index, computed without a signed left shift.
  std::int64_t contribution{0}; ///< digit times scaled_multiplicand.
  std::int64_t accumulator{0}; ///< Sum after this contribution.
};
/// @brief Owned operands, exact result/counts and the 32 canonical recoding decisions.
/// Recording allocates O(32) entries. The ordinary function uses the same loop
/// without allocating a trace. Data are owned; independent calls share no state.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::booth_multiply_trace(-7,6);
/// std::print("{} {}\n",trace.steps.size(),trace.result.product); // 32 -42
/// @endcode
/// @see booth_multiply_trace(), BoothStep
struct BoothTrace {
  std::int32_t multiplicand{0}; ///< Original signed multiplicand.
  std::int32_t multiplier{0}; ///< Original signed multiplier.
  BoothResult result; ///< Exact product and add/subtract counts from the same traversal.
  std::vector<BoothStep> steps; ///< Decisions in bit-zero-to-bit-31 order.
};
/// @brief Multiply any two int32 values exactly by 32 radix-2 Booth decisions.
/// At bit i the signed digit is b_(i-1)-b_i, with b_-1=0. Both current and
/// preceding one means no operation; (1,0) subtracts and (0,1) adds. The final
/// sign bit supplies the two's-complement negative weight without a 33rd step.
/// Widened int64 products/sums stay representable, including INT32_MIN operands.
/// Exactly 32 decisions, O(1) storage, no allocation or mutation of caller/shared data.
/// Fewer nonzero digits for long runs is not a runtime or hardware speed guarantee.
/// @param multiplicand Any signed 32-bit value, including INT32_MIN.
/// @param multiplier Any signed 32-bit value; its fixed-width representation drives counts.
/// @return Exact int64 product plus numbers of +1 and -1 recoding digits.
/// @par Example
/// @code
/// const auto result=dm::algorithms::booth_multiply(123456,-789);
/// std::print("{}\n",result.product); // -97406784
/// @endcode
/// @par References
/// - A. D. Booth (1951), A signed binary multiplication technique, Quarterly Journal
///   of Mechanics and Applied Mathematics 4(2), 236-240, doi:10.1093/qjmam/4.2.236.
///   This implementation accumulates widened shifted terms rather than emulating registers.
/// @see booth_multiply_trace(), BoothResult
BoothResult booth_multiply(std::int32_t multiplicand,std::int32_t multiplier);
/// @brief Record the exact same 32 Booth decisions used by booth_multiply().
/// Includes zero digits, widened terms and post-decision accumulators. Time and
/// owned recording storage are O(32); no input or shared state is mutated.
/// @param multiplicand Any signed int32 value.
/// @param multiplier Any signed int32 value, recoded in its full 32-bit representation.
/// @return Owned operands, result/counts and every canonical decision.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::booth_multiply_trace(3,7);
/// std::print("{} {}\n",trace.steps[3].contribution,trace.result.product); // 24 21
/// @endcode
/// @par References
/// - A. D. Booth (1951), A signed binary multiplication technique,
///   doi:10.1093/qjmam/4.2.236; widened signed-digit accumulation here.
/// @see booth_multiply(), BoothTrace
BoothTrace booth_multiply_trace(std::int32_t multiplicand,std::int32_t multiplier);
} // namespace datamunge::algorithms
