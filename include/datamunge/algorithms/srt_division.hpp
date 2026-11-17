#pragma once
/// @file srt_division.hpp
/// @brief Checked floating radix-two redundant quotient transitions.
#include <array>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SRTOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SRTOptions r{};
/// (void)r;
/// @endcode
/// @see srt_division_run()
struct SRTOptions {
  std::size_t max_bits{4096};       ///< Complete requested digit count; zero disables policy.
  std::size_t max_records{1000000}; ///< Captured actual step records; zero disables.
};
/// @brief Own the SRTStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SRTStep r{};
/// (void)r;
/// @endcode
/// @see srt_division_run()
struct SRTStep {
  std::size_t index{};   ///< Actual zero-based fractional digit position.
  double      before{};  ///< Actual normalized partial remainder before shifting.
  double      shifted{}; ///< Actual two-times partial remainder used for selection.
  int         digit{};   ///< Actual selected redundant digit-1,0,+1.
  double      after{};   ///< Actual shifted remainder minus digit times normalized divisor.
  double      weight{};  ///< Actual double2^-(index+1), possibly underflowing for very long runs.
  double      value{};   ///< Actual accumulated floating quotient prefix.
};
/// @brief Own the SRTRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SRTRun r{};
/// (void)r;
/// @endcode
/// @see srt_division_run()
struct SRTRun {
  int normalization_exponent{}; ///< Actual power-of-two scale applied to original numerator/divisor.
  double
      normalized_numerator{}; ///< Actual ldexp numerator; may underflow for extreme input ratios.
  double           normalized_divisor{}; ///< Actual divisor in[0.5,1).
  std::vector<int> digits;               ///< Actual complete redundant quotient stream.
  double           value{};              ///< Actual reconstructed double quotient prefix.
  double           remainder{};          ///< Actual final normalized partial remainder.
  double
      nominal_tail_bound{}; ///< Double representation of mathematical2^-bits; may underflow to0, not a floating-error certificate.
  std::array<std::size_t, 3> counts{}; ///< Actual emitted counts for digits-1,0,+1.
  std::vector<SRTStep>       steps;    ///< Optional actual producing remainder/accumulator states.
};
/// @brief Compute proper-fraction radix-two SRT quotient digits.
/// Requires finite D>0 and0<=N<D, nonnegative bit count. frexp selects d in[1/2,1); n=ldexp(N,e). Each step sets w=2P, q=+1 if w>=1/2,-1 if w<=-1/2, otherwise0, then P=w-q*d and accumulates q*2^-j. Exact arithmetic identity n/d=Q_j+P_j/(d*2^j), with |P_j|<d, gives truncation error<2^-j. This double simulation has normalization/recurrence/accumulation roundoff, possible normalized-numerator or weight underflow and no certified error claim beyond machine precision. It is not a hardware leading-bit selection table or a timing model. O(bits) time/output storage, optional O(bits) histories; complete record geometry checked first. Owned results, no shared state.
/// @param N Complete finite proper-fraction numerator.
/// @param D Complete finite strictly positive divisor greater than N.
/// @param bits Nonnegative requested redundant fractional digit count, default53.
/// @param options Complete digit/capture policies; zero disables policies.
/// @return Owned actual normalized stream, quotient, remainder, counts and optional history.
/// @throws std::invalid_argument Malformed inputs or negative bit count.
/// @throws std::length_error Digit/history/counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::srt_division_run(.5,1.,4);
/// std::print("{}\n",r.value); // 0.5
/// @endcode
/// @par References
/// - J. E. Robertson, A New Class of Digital Division Methods, IRE Trans. Electronic Computers EC-7(3),1958; redundant-digit recurrence, not a physical implementation.
/// @see srt_divide()
SRTRun srt_division_run(double N, double D, int bits = 53, const SRTOptions& options = {});
/// @brief Compute proper-fraction radix-two SRT quotient digits with actual transitions.
/// Requires finite D>0 and0<=N<D, nonnegative bit count. frexp selects d in[1/2,1); n=ldexp(N,e). Each step sets w=2P, q=+1 if w>=1/2,-1 if w<=-1/2, otherwise0, then P=w-q*d and accumulates q*2^-j. Exact arithmetic identity n/d=Q_j+P_j/(d*2^j), with |P_j|<d, gives truncation error<2^-j. This double simulation has normalization/recurrence/accumulation roundoff, possible normalized-numerator or weight underflow and no certified error claim beyond machine precision. It is not a hardware leading-bit selection table or a timing model. O(bits) time/output storage, optional O(bits) histories; complete record geometry checked first. Owned results, no shared state.
/// @param N Complete finite proper-fraction numerator.
/// @param D Complete finite strictly positive divisor greater than N.
/// @param bits Nonnegative requested redundant fractional digit count, default53.
/// @param options Complete digit/capture policies; zero disables policies.
/// @return Owned actual normalized stream, quotient, remainder, counts and optional history.
/// @throws std::invalid_argument Malformed inputs or negative bit count.
/// @throws std::length_error Digit/history/counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::srt_division_trace(.5,1.,4);
/// std::print("{}\n",r.value); // 0.5
/// @endcode
/// @par References
/// - J. E. Robertson, A New Class of Digital Division Methods, IRE Trans. Electronic Computers EC-7(3),1958; redundant-digit recurrence, not a physical implementation.
/// @see srt_divide()
SRTRun srt_division_trace(double N, double D, int bits = 53, const SRTOptions& options = {});
/// @brief Return the preserved checked redundant SRT digit stream.
/// Same actual producing recurrence and default policies.
/// @param N Finite0<=N<D.
/// @param D Finite strictly positive divisor.
/// @param bits Nonnegative fractional digit count.
/// @return Owned digits, each-1,0,+1.
/// @throws std::invalid_argument Malformed proper fraction/count.
/// @throws std::length_error Default digit/capacity bound exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::srt_quotient_digits(.5,1.,4)); // [1, 0, 0, 0]
/// @endcode
/// @see srt_division_run()
std::vector<int> srt_quotient_digits(double N, double D, int bits);
/// @brief Reconstruct a checked redundant fractional stream in double.
/// Uses the same weighted accumulator as the producing recurrence. Validates every digit in{-1,0,1}; weights may underflow after1074 and rounding may dominate before that. No shared state, O(digits) time and constant extra working storage. Default digit bound4096.
/// @param digits Complete borrowed redundant fractional digit stream.
/// @return Actual floating weighted sum.
/// @throws std::invalid_argument Digit outside-1,0,1.
/// @throws std::length_error More than4096 digits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::srt_digits_to_value({1,0,0})); // 0.5
/// @endcode
/// @see srt_division_run()
double srt_digits_to_value(const std::vector<int>& digits);
/// @brief Return the preserved checked floating SRT quotient.
/// Same actual producer and default policies; beyond-double-precision digits do not certify more accurate output.
/// @param N Finite0<=N<D.
/// @param D Finite strictly positive divisor.
/// @param bits Nonnegative fractional digit count, default53.
/// @return Actual reconstructed double prefix.
/// @throws std::invalid_argument Malformed proper fraction/count.
/// @throws std::length_error Default digit/capacity bound exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::srt_divide(.5,1.,4)); // 0.5
/// @endcode
/// @see srt_division_run()
double srt_divide(double N, double D, int bits = 53);
} // namespace datamunge::algorithms
