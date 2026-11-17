#pragma once
/// @file newton_division.hpp
/// @brief Checked scaled floating division with actual shared reciprocal-Newton histories.
#include <cstdint>
#include <datamunge/algorithms/newton_reciprocal.hpp>
#include <utility>
namespace datamunge::algorithms {
/// @brief Actual scaled floating quotient and reciprocal stopping metadata.
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_division_run(1,2);std::print("{}\n",r.quotient); // 0.5
/// @endcode
/// @see newton_division_trace()
struct NewtonDivisionRun {
  double quotient{0}; ///< Actual normalized mantissa product rescaled by exponent difference.
  double numerator_mantissa{0}; ///< Actual frexp(abs(numerator)) mantissa; zero in zero shortcut.
  double denominator_mantissa{
      0};                       ///< Actual frexp(abs(denominator)) mantissa; zero in zero shortcut.
  int    numerator_exponent{0}; ///< Actual numerator frexp exponent; zero in zero shortcut.
  int    denominator_exponent{0}; ///< Actual denominator frexp exponent; zero in zero shortcut.
  double seed{0};       ///< Actual48/17−(32/17)*denominator_mantissa; zero in zero shortcut.
  double reciprocal{0}; ///< Actual final normalized-denominator reciprocal; zero in zero shortcut.
  std::size_t updates{0};        ///< Actual shared double reciprocal updates, at most60.
  bool        stationary{false}; ///< Actual abs(next-before)<=1e-17*abs(next) stopping test passed.
  bool zero_shortcut{false};     ///< Original numerator was signed zero; no reciprocal updates ran.
  bool underflowed{false};       ///< Nonzero numerator produced rounded quotient zero.
  bool subnormal{false};         ///< Actual quotient is nonzero subnormal.
};
/// @brief Owned original scalar inputs and actual normalized reciprocal updates.
/// @par Example
/// @code
/// auto t=dm::algorithms::newton_division_trace(0,2);std::print("{}\n",t.captured_entries); // 3
/// @endcode
/// @see newton_division_trace()
struct NewtonDivisionTrace {
  double            numerator{0};          ///< Original finite numerator, signed zero preserved.
  double            denominator{0};        ///< Original finite nonzero denominator.
  NewtonDivisionRun run;                   ///< Actual result and scaling/stopping metadata.
  std::vector<NewtonReciprocalStep> steps; ///< Actual normalized reciprocal updates, no replay.
  std::size_t                       captured_entries{
      0}; ///< Actual logical history units3+updates; fixed run metadata excluded.
};
/// @brief Approximate a finite quotient by scaled double reciprocal-Newton multiplication.
/// Both operands must be finite; denominator must be nonzero. Nonzero magnitudes
/// are separately normalized with frexp. Seed48/17−(32/17)*m_D then updates
/// x<-x*(2-m_D*x), stopping at relative state change<=1e-17 or60 updates.
/// Final ldexp(sign*m_N*x,e_N-e_D) avoids overflowing an intermediate raw
/// reciprocal when the quotient is representable, including equal subnormals.
/// Signed-zero numerator takes a signed-zero quotient shortcut. Actual quotient
/// overflow throws; gradual underflow/rounded zero are permitted and reported.
/// This is a double approximation, not a correctly-rounded division guarantee,
/// hardware speed claim or constant-time primitive. Seed constants/rescaling
/// do not establish a division-free CPU instruction stream. O(60) fixed work,
/// constant driver storage; no input/shared state mutation.
/// @param numerator Finite original dividend.
/// @param denominator Finite original nonzero divisor, either sign.
/// @return Actual double quotient/result, with stated IEEE underflow conventions.
/// @throws std::invalid_argument If complete operands are invalid before arithmetic.
/// @throws std::overflow_error If an update or final quotient is nonfinite.
/// @throws std::length_error If actual history policy is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_raphson_division(-7,1024);std::print("{}\n",r); // -0.0068359375
/// @endcode
/// @see NewtonDivisionRun, NewtonDivisionTrace, newton_reciprocal_trace()
double newton_raphson_division(double numerator, double denominator);
/// @brief Approximate a finite quotient by scaled double reciprocal-Newton multiplication.
/// Both operands must be finite; denominator must be nonzero. Nonzero magnitudes
/// are separately normalized with frexp. Seed48/17−(32/17)*m_D then updates
/// x<-x*(2-m_D*x), stopping at relative state change<=1e-17 or60 updates.
/// Final ldexp(sign*m_N*x,e_N-e_D) avoids overflowing an intermediate raw
/// reciprocal when the quotient is representable, including equal subnormals.
/// Signed-zero numerator takes a signed-zero quotient shortcut. Actual quotient
/// overflow throws; gradual underflow/rounded zero are permitted and reported.
/// This is a double approximation, not a correctly-rounded division guarantee,
/// hardware speed claim or constant-time primitive. Seed constants/rescaling
/// do not establish a division-free CPU instruction stream. O(60) fixed work,
/// constant driver storage; no input/shared state mutation.
/// @param numerator Finite original dividend.
/// @param denominator Finite original nonzero divisor, either sign.
/// @return Actual NewtonDivisionRun quotient/result, with stated IEEE underflow conventions.
/// @throws std::invalid_argument If complete operands are invalid before arithmetic.
/// @throws std::overflow_error If an update or final quotient is nonfinite.
/// @throws std::length_error If actual history policy is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_division_run(-7,1024);std::print("{}\n",r.quotient); // -0.0068359375
/// @endcode
/// @see NewtonDivisionRun, NewtonDivisionTrace, newton_reciprocal_trace()
NewtonDivisionRun newton_division_run(double numerator, double denominator);
/// @brief Capture actual scaled floating Newton division and reciprocal updates.
/// Both operands must be finite; denominator must be nonzero. Nonzero magnitudes
/// are separately normalized with frexp. Seed48/17−(32/17)*m_D then updates
/// x<-x*(2-m_D*x), stopping at relative state change<=1e-17 or60 updates.
/// Final ldexp(sign*m_N*x,e_N-e_D) avoids overflowing an intermediate raw
/// reciprocal when the quotient is representable, including equal subnormals.
/// Signed-zero numerator takes a signed-zero quotient shortcut. Actual quotient
/// overflow throws; gradual underflow/rounded zero are permitted and reported.
/// This is a double approximation, not a correctly-rounded division guarantee,
/// hardware speed claim or constant-time primitive. Seed constants/rescaling
/// do not establish a division-free CPU instruction stream. O(60) fixed work,
/// constant driver storage plus actual owned trace records; no input/shared state mutation.
/// @param numerator Finite original dividend.
/// @param denominator Finite original nonzero divisor, either sign.
/// @param max_entries Actual3+updates history cap, default1000000, zero disables; checked before each shared update.
/// @return Owned original inputs and actual same-producer history.
/// @throws std::invalid_argument If complete operands are invalid before arithmetic.
/// @throws std::overflow_error If an update or final quotient is nonfinite.
/// @throws std::length_error If actual history policy is exceeded before the next update.
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_division_trace(-7,1024);std::print("{}\n",r.run.quotient); // -0.0068359375
/// @endcode
/// @see NewtonDivisionRun, NewtonDivisionTrace, newton_reciprocal_trace()
NewtonDivisionTrace
newton_division_trace(double numerator, double denominator, std::size_t max_entries = 1000000);
/// @brief Return the original four signed floating-division worked pairs.
/// @return Owned pairs(1,3),(-7,1024),(5,1e-9),(22,7), in original order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newton_division_demo_pairs().size()); // 4
/// @endcode
/// @see newton_raphson_division()
std::vector<std::pair<double, double>> newton_division_demo_pairs();
/// @brief Actual finite sampled comparison with native built-in division, not a mathematical error bound.
/// @par Example
/// @code
/// dm::algorithms::NewtonDivisionCheck c;std::print("{}\n",c.checked); // 0
/// @endcode
/// @see newton_division_trace()
struct NewtonDivisionCheck {
  std::size_t checked{0};       ///< Actual generated comparisons.
  double max_relative_error{0}; ///< Largest abs((native_approx-built_in)/built_in), evaluated wide.
  std::size_t worst_index{
      0}; ///< First actual index giving the strictly largest observed error, zero if all equal.
};
/// @brief Perform the originally claimed100000-case sampled comparison as an actual native study.
/// Each pair draws two top53-bit unit words, two exponent words modulo1001
/// shifted by-500 and two sign words, in that order, from local MT19937_64.
/// Mantissas are .5+.5*u; quotients remain finite/nonzero in this bounded range.
/// Calls the canonical division producer and compares with built-in double /.
/// O(samples*60) work, constant driver storage, no global RNG state mutation.
/// @param samples Actual positive/zero comparison budget, default100000, at most1000000.
/// @param seed Local MT19937_64 seed, default7.
/// @return Actual sampled maximum relative difference and first worst index.
/// @throws std::length_error If the explicit study budget cap is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newton_division_check(10).checked); // 10
/// @endcode
/// @see NewtonDivisionCheck
NewtonDivisionCheck newton_division_check(std::size_t samples = 100000, std::uint64_t seed = 7);
} // namespace datamunge::algorithms
