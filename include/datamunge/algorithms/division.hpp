#pragma once
/// @file division.hpp
/// @brief Checked full-width unsigned division, actual register/reciprocal decisions and explicit legacy zero sentinel.
#include <cstdint>
#include <datamunge/algorithms/newton_reciprocal.hpp>
#include <string>
#include <utility>
namespace datamunge::algorithms {
/// @brief Legacy field-order quotient/remainder and algorithm-specific iteration count.
/// Divisor0 retains the historical{0,0,0}sentinel, which is NOT a defined
/// quotient/remainder. Use analyzed/trace valid_divisor to distinguish it.
/// @par Example
/// @code
/// auto r=dm::algorithms::non_restoring_divide(217,13);std::print("{} {}\n",r.quotient,r.remainder); // 16 9
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct DivResultEx {
  std::uint64_t quotient{0};  ///< Exact quotient for nonzero divisor.
  std::uint64_t remainder{0}; ///< Exact remainder below nonzero divisor.
  int           iterations{
      0}; ///< 64 bit steps for restoring/nonrestoring;5 reciprocal updates for nontrivial Newton.
};
/// @brief Exact signed magnitude using two unsigned64-bit words.
/// @par Example
/// @code
/// dm::algorithms::DivisionWideValue v{0,17,true};std::print("{}\n",dm::algorithms::division_wide_decimal(v)); // -17
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct DivisionWideValue {
  std::uint64_t high{0};         ///< High64 magnitude bits.
  std::uint64_t low{0};          ///< Low64 magnitude bits.
  bool          negative{false}; ///< Magnitude sign; signed zero displays0.
};
/// @brief Convert an exact signed magnitude to a finite double for plotting only.
/// No integer decision or certificate uses this rounded display value.
/// @param value Complete128-bit magnitude/sign, all bit patterns allowed.
/// @return Rounded finite floating approximation with the supplied sign.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::division_wide_approximation({0,17,true})); // -17
/// @endcode
/// @see division_wide_decimal()
double division_wide_approximation(DivisionWideValue value);
/// @brief Format an owned exact signed magnitude as ordinary decimal text.
/// Pure formatting may divide by10; those operations are outside division-producer counters.
/// @param value Full128-bit magnitude and sign, all bit patterns allowed.
/// @return Decimal sign/digits without exponent or leading zeros, negative zero becomes0.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::division_wide_decimal({1,0,false})); // 18446744073709551616
/// @endcode
/// @see DivisionWideValue
std::string division_wide_decimal(DivisionWideValue value);
/// @brief Actual exact result and selected producer work, excluding formatting/guards/loop control.
/// @par Example
/// @code
/// auto a=dm::algorithms::restoring_division_analyze(217,13);std::print("{} {}\n",a.bit_steps,a.subtractions); // 64 1
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct DivisionAnalysis {
  DivResultEx result; ///< Actual legacy-layout result, including zero sentinel.
  bool        valid_divisor{
      false}; ///< False only for original divisor0; no mathematical division performed then.
  std::size_t bit_steps{0}; ///< Actual64 register updates for restoring/nonrestoring,0 for Newton.
  std::size_t comparison_tests{
      0}; ///< Actual restoring magnitude tests, nonrestoring sign tests inclfinal, or Newton exact correction tests.
  std::size_t subtractions{
      0}; ///< Actual restoring conditional subtracts or nonrestoring subtract branches.
  std::size_t additions{
      0}; ///< Actual nonrestoring add branches and optional final correction;0 for comparison-first restoring.
  std::size_t reciprocal_updates{
      0}; ///< Actual shared double reciprocal updates, five for nontrivial Newton.
  std::size_t decrements{0}; ///< Actual Newton exact downward quotient corrections.
  std::size_t increments{0}; ///< Actual Newton exact upward quotient corrections.
  std::size_t hypothetical_restores{
      0}; ///< Restoring failed comparisons, a hypothetical trial-subtract/add-back count, not executed additions.
};
/// @brief Actual comparison-first unsigned remainder/quotient bit step.
/// @par Example
/// @code
/// auto t=dm::algorithms::restoring_division_trace(217,13);std::print("{}\n",t.steps.size()); // 64
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct RestoringDivisionStep {
  int           bit{0};              ///< Actual original dividend bit index,63 down to0.
  std::uint64_t remainder_before{0}; ///< Actual remainder before appending this dividend bit.
  std::uint64_t shifted{0}; ///< Actual shifted remainder plus incoming original dividend bit.
  std::uint64_t remainder_after{0}; ///< Actual remainder after conditional subtraction.
  std::uint64_t quotient_before{0}; ///< Actual quotient before this bit.
  std::uint64_t quotient_after{0};  ///< Actual quotient after committing this bit.
  bool          subtracted{
      false}; ///< Actual shifted>=divisor comparison result; false executes no add-back.
};
/// @brief Owned original inputs and all actual comparison-first bit steps.
/// @par Example
/// @code
/// auto t=dm::algorithms::restoring_division_trace(217,13);std::print("{}\n",t.captured_entries); // 69
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct RestoringDivisionTrace {
  std::uint64_t    dividend{0}; ///< Original unsigned dividend.
  std::uint64_t    divisor{0};  ///< Original unsigned divisor,0 sentinel permitted.
  DivisionAnalysis analysis;    ///< Actual same-producer result/work.
  std::vector<RestoringDivisionStep>
              steps; ///< Actual64 bit records for nonzero divisor, empty for zero sentinel.
  std::size_t captured_entries{0}; ///< Complete logical units5+actual bit records.
};
/// @brief Actual signed remainder and64-bit quotient-register transition.
/// @par Example
/// @code
/// auto t=dm::algorithms::non_restoring_division_trace(217,13);std::print("{}\n",t.steps.size()); // 64
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct NonRestoringDivisionStep {
  int           bit{0}; ///< Actual original bit position,63 down to0.
  std::uint64_t quotient_before{
      0}; ///< Actual Q register before shift, initially original dividend.
  std::uint64_t     quotient_after{0}; ///< Actual modulo2^64 Q register after shift/output bit.
  bool              incoming{false};   ///< Actual old Q most-significant bit transferred to A.
  DivisionWideValue before;            ///< Actual signed A before shift.
  DivisionWideValue shifted;           ///< Actual exact A*2+incoming before add/subtract.
  DivisionWideValue after;             ///< Actual signed A after one add/subtract.
  bool subtracted{false}; ///< True subtract divisor, false add divisor, selected by shifted sign.
  bool output_bit{false}; ///< Actual new A>=0 quotient low bit.
};
/// @brief Owned original inputs, all actual register steps and actual final correction.
/// @par Example
/// @code
/// auto t=dm::algorithms::non_restoring_division_trace(217,13);std::print("{}\n",t.captured_entries); // 70
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct NonRestoringDivisionTrace {
  std::uint64_t    dividend{0}; ///< Original unsigned dividend.
  std::uint64_t    divisor{0};  ///< Original unsigned divisor,0 sentinel permitted.
  DivisionAnalysis analysis;    ///< Actual same-producer result/work.
  std::vector<NonRestoringDivisionStep>
                    steps; ///< Actual64 exact signed-register transitions, empty for zero sentinel.
  DivisionWideValue final_before; ///< Actual A after final bit, before any final correction.
  DivisionWideValue final_after;  ///< Actual A after optional final divisor addition.
  bool              final_corrected{
      false}; ///< Whether a negative final A actually required one additional addition.
  std::size_t captured_entries{0}; ///< Complete logical units6+actual bit records.
};
/// @brief Actual successful or unsuccessful exact quotient-correction comparison.
/// @par Example
/// @code
/// auto t=dm::algorithms::integer_newton_division_trace(100,7);std::print("{}\n",t.checks.empty()); // false
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct IntegerNewtonCheck {
  std::uint64_t     quotient_before{0}; ///< Actual candidate quotient before this comparison.
  DivisionWideValue tested_product;     ///< Exact q*D for downward test or(q+1)*D for upward test.
  bool downward{false}; ///< True downward greater-than-N test, false upward at-most-N test.
  bool changed{false};  ///< Whether this exact comparison actually changes the quotient.
  std::uint64_t quotient_after{0}; ///< Actual candidate quotient after the comparison.
};
/// @brief Owned original operands and actual floating-reciprocal/exact-correction decisions.
/// @par Example
/// @code
/// auto t=dm::algorithms::integer_newton_division_trace(5,0);std::print("{}\n",t.analysis.valid_divisor); // false
/// @endcode
/// @see non_restoring_division_trace(), integer_newton_division_trace()
struct IntegerNewtonDivisionTrace {
  std::uint64_t    dividend{0};            ///< Original unsigned dividend.
  std::uint64_t    divisor{0};             ///< Original unsigned divisor,0 sentinel permitted.
  DivisionAnalysis analysis;               ///< Actual same-producer exact result/work.
  double           rounded_denominator{0}; ///< Actual double(divisor),0 placeholder for shortcuts.
  double           seed{0};       ///< Actual1.0/rounded_denominator seed: uses a floating divide.
  double           reciprocal{0}; ///< Actual reciprocal after five shared double updates.
  double           quotient_estimate{
      0}; ///< Actual double(dividend)*reciprocal before guarded integer conversion.
  std::uint64_t initial_quotient{0}; ///< Actual safely clamped/truncated initial integer candidate.
  bool          estimate_clamped{
      false}; ///< Actual estimate>=2^64 guard clamped it to UINT64_MAX before casting.
  std::vector<NewtonReciprocalStep> steps; ///< Actual five reciprocal updates, empty for shortcuts.
  std::vector<IntegerNewtonCheck>
      checks; ///< Every actual exact downward/upward comparison, including both final false checks.
  std::size_t captured_entries{
      0}; ///< Actual logical units10+reciprocal records+exact comparison records.
};
/// @brief Compute actual comparison-first restoring unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// Always64 original-bit steps. Compare shifted remainder with D, subtract only
/// when it fits: this software executes NO restoration additions. Failed
/// comparisons count only a separately labeled hypothetical trial/restore model.
/// O(64) work, constant driver state plus optional owned64-record history.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @return Owned actual DivResultEx with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::restoring_divide(100,7);std::print("{} {}\n",r.quotient,r.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
DivResultEx restoring_divide(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Compute actual comparison-first restoring unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// Always64 original-bit steps. Compare shifted remainder with D, subtract only
/// when it fits: this software executes NO restoration additions. Failed
/// comparisons count only a separately labeled hypothetical trial/restore model.
/// O(64) work, constant driver state plus optional owned64-record history.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @return Owned actual DivisionAnalysis with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::restoring_division_analyze(100,7);std::print("{} {}\n",r.result.quotient,r.result.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
DivisionAnalysis restoring_division_analyze(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Capture actual comparison-first restoring unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// Always64 original-bit steps. Compare shifted remainder with D, subtract only
/// when it fits: this software executes NO restoration additions. Failed
/// comparisons count only a separately labeled hypothetical trial/restore model.
/// O(64) work, constant driver state plus optional owned64-record history.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @param max_entries Logical complete/actual history cap, default1000000, zero disables.
/// @return Owned actual RestoringDivisionTrace with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry/quota is exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::restoring_division_trace(100,7);std::print("{} {}\n",r.analysis.result.quotient,r.analysis.result.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
RestoringDivisionTrace restoring_division_trace(
    std::uint64_t dividend,
    std::uint64_t divisor,
    std::size_t   max_entries = 1000000);
/// @brief Compute actual signed non-restoring unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// Always64 steps of exact signed A*2+incoming followed by one add/subtract;
/// multiplication avoids negative signed-shift hazards. Modulo64-bit Q shifts
/// model the register explicitly. Add D once more if final A is negative.
/// O(64) work, constant driver state plus optional owned64-record history.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @return Owned actual DivResultEx with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::non_restoring_divide(100,7);std::print("{} {}\n",r.quotient,r.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
DivResultEx non_restoring_divide(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Compute actual signed non-restoring unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// Always64 steps of exact signed A*2+incoming followed by one add/subtract;
/// multiplication avoids negative signed-shift hazards. Modulo64-bit Q shifts
/// model the register explicitly. Add D once more if final A is negative.
/// O(64) work, constant driver state plus optional owned64-record history.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @return Owned actual DivisionAnalysis with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::non_restoring_division_analyze(100,7);std::print("{} {}\n",r.result.quotient,r.result.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
DivisionAnalysis non_restoring_division_analyze(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Capture actual signed non-restoring unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// Always64 steps of exact signed A*2+incoming followed by one add/subtract;
/// multiplication avoids negative signed-shift hazards. Modulo64-bit Q shifts
/// model the register explicitly. Add D once more if final A is negative.
/// O(64) work, constant driver state plus optional owned64-record history.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @param max_entries Logical complete/actual history cap, default1000000, zero disables.
/// @return Owned actual NonRestoringDivisionTrace with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry/quota is exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::non_restoring_division_trace(100,7);std::print("{} {}\n",r.analysis.result.quotient,r.analysis.result.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
NonRestoringDivisionTrace non_restoring_division_trace(
    std::uint64_t dividend,
    std::uint64_t divisor,
    std::size_t   max_entries = 1000000);
/// @brief Compute actual double-reciprocal plus exact-correction unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// N<D returns0,N with0 iterations. Otherwise convert D to double, seed1/d
/// with an actual floating divide, and execute FIVE double Newton updates.
/// This is not a fixed-point/division-free whole-call implementation. Guard
/// estimate>=2^64 before integer conversion; exact unsigned128 comparisons
/// correct q downward/upward. Promote q BEFORE adding1 to avoid UINT64 wrap.
/// O(5+corrections) word work, not a constant-time or hardware-speed guarantee;
/// optional history retains every successful/unsuccessful exact comparison.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @return Owned actual DivResultEx with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_raphson_divide(100,7);std::print("{} {}\n",r.quotient,r.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
DivResultEx newton_raphson_divide(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Compute actual double-reciprocal plus exact-correction unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// N<D returns0,N with0 iterations. Otherwise convert D to double, seed1/d
/// with an actual floating divide, and execute FIVE double Newton updates.
/// This is not a fixed-point/division-free whole-call implementation. Guard
/// estimate>=2^64 before integer conversion; exact unsigned128 comparisons
/// correct q downward/upward. Promote q BEFORE adding1 to avoid UINT64 wrap.
/// O(5+corrections) word work, not a constant-time or hardware-speed guarantee;
/// optional history retains every successful/unsuccessful exact comparison.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @return Owned actual DivisionAnalysis with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry is exceeded (trace entry point only).
/// @par Example
/// @code
/// auto r=dm::algorithms::integer_newton_division_analyze(100,7);std::print("{} {}\n",r.result.quotient,r.result.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
DivisionAnalysis integer_newton_division_analyze(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Capture actual double-reciprocal plus exact-correction unsigned64-bit division.
/// Nonzero divisor returns exact q,r with N=q*D+r and0<=r<D. Divisor0
/// preserves the legacy{0,0,0}sentinel; analyzed valid_divisor=false makes
/// clear it is NOT mathematical division. Empty histories accompany shortcuts.
/// N<D returns0,N with0 iterations. Otherwise convert D to double, seed1/d
/// with an actual floating divide, and execute FIVE double Newton updates.
/// This is not a fixed-point/division-free whole-call implementation. Guard
/// estimate>=2^64 before integer conversion; exact unsigned128 comparisons
/// correct q downward/upward. Promote q BEFORE adding1 to avoid UINT64 wrap.
/// O(5+corrections) word work, not a constant-time or hardware-speed guarantee;
/// optional history retains every successful/unsuccessful exact comparison.
/// No input mutation/caller state retained; unsigned wrap is only intentional
/// quotient-register behavior. Exact product checks avoid signed128 overflow.
/// @param dividend Original unsigned64-bit N.
/// @param divisor Original unsigned64-bit D,0 legacy sentinel explicitly retained.
/// @param max_entries Logical complete/actual history cap, default1000000, zero disables.
/// @return Owned actual IntegerNewtonDivisionTrace with explicit zero-divisor convention.
/// @throws std::length_error If optional history count geometry/quota is exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::integer_newton_division_trace(100,7);std::print("{} {}\n",r.analysis.result.quotient,r.analysis.result.remainder); // 14 2
/// @endcode
/// @see DivisionAnalysis
IntegerNewtonDivisionTrace integer_newton_division_trace(
    std::uint64_t dividend,
    std::uint64_t divisor,
    std::size_t   max_entries = 1000000);
/// @brief Return the original100/7,1000000/999,123456789/12345,UINT64_MAX/7 examples.
/// @return Owned original-order four unsigned operand pairs.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::integer_newton_demo_pairs().size()); // 4
/// @endcode
/// @see integer_newton_division_trace(), non_restoring_division_trace()
std::vector<std::pair<std::uint64_t, std::uint64_t>> integer_newton_demo_pairs();
/// @brief Return the original217/13,100/7,1000000/999,UINT64_MAX/3 examples.
/// @return Owned original-order four unsigned operand pairs.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nonrestoring_demo_pairs().size()); // 4
/// @endcode
/// @see integer_newton_division_trace(), non_restoring_division_trace()
std::vector<std::pair<std::uint64_t, std::uint64_t>> nonrestoring_demo_pairs();
/// @brief Return the original six D=13 work-study dividends, without simulating a different algorithm.
/// @return Owned217,4096,65535,1000000,16777215,4294967295 values.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::division_work_dividends().size()); // 6
/// @endcode
/// @see restoring_division_analyze(), non_restoring_division_analyze()
std::vector<std::uint64_t> division_work_dividends();
} // namespace datamunge::algorithms
