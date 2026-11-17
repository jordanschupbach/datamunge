#pragma once
/// @file lucas_primality.hpp
/// @brief Exact Lucas order certificates with actual factor/base/power evidence and explicit inconclusive budgets.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual conclusive filter/certificate or explicit inconclusive work limit.
/// @par Example
/// @code
/// auto a=dm::algorithms::lucas_primality_analyze(7);
/// std::print("{}\n",a.stop==dm::algorithms::LucasPrimalityStop::PrimeCertificate); // true
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
enum class LucasPrimalityStop {
  SmallComposite, ///< Input0 or1, conclusive nonprime by definition.
  SmallPrime,     ///< Input2, conclusive prime without factoring.
  EvenComposite,  ///< Even input greater than2, conclusive composite.
  FactorBudget, ///< Trial-divisor budget exhausted before complete n-minus1 factorization; inconclusive.
  BaseBudget,      ///< Base budget exhausted before a certificate or Fermat failure; inconclusive.
  CompositeFermat, ///< Actual base violates a^(n-minus1)=1 modulo n, proving compositeness.
  PrimeCertificate ///< Actual base has exact multiplicative order n-minus1, proving primality.
};
/// @brief One actual distinct prime factor and its complete extracted multiplicity.
/// @par Example
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(7);
/// std::print("{}\n",t.analysis.factors.front().prime); // 2
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasPrimePower {
  std::uint64_t
      prime; ///< Proven prime trial factor, or proven final residual after completed trial division.
  unsigned exponent; ///< Positive actual extracted multiplicity.
};
/// @brief Actual selected complete-factorization and witness-search work.
/// @par Example
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(7);
/// std::print("{}\n",t.analysis.work.bases); // 2
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasPrimalityWork {
  std::size_t trial_divisors{0}; ///< Actual candidate divisor tests, including nonfactors.
  std::size_t successful_divisions{
      0}; ///< Actual exact factor-removal divisions, excluding final residual recognition.
  std::size_t bases{0}; ///< Actual examined bases, in increasing order starting at2.
  std::size_t powers{
      0}; ///< Actual canonical modular-exponentiation calls, excluding no hidden replay.
};
/// @brief Actual bounded Lucas result; only conclusive outcomes decide primality/compositeness.
/// @par Example
/// @code
/// auto a=dm::algorithms::lucas_primality_analyze(7);
/// std::print("{} {}\n",a.prime,a.witness); // true 3
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasPrimalityAnalysis {
  std::uint64_t      n{0}; ///< Original unsigned candidate.
  LucasPrimalityStop stop{
      LucasPrimalityStop::SmallComposite}; ///< Actual explicit completion or exhaustion cause.
  bool conclusive{false}; ///< True only for a mathematical prime/composite conclusion.
  bool prime{
      false}; ///< True only with a conclusive prime conclusion; false alone is not a composite certificate.
  bool factorization_complete{
      false}; ///< True only after actual complete n-minus1 factorization, or trivial n2 case.
  std::uint64_t remaining{
      0}; ///< Unfactored residual on FactorBudget; one after completed factorization.
  std::vector<LucasPrimePower>
      factors; ///< Actual extracted sorted distinct prime powers, partial if factor budget exhausted.
  std::uint64_t witness{0}; ///< Actual least tested Lucas order witness;1 for n2, zero when absent.
  std::uint64_t composite_base{
      0}; ///< Actual Fermat-failing base when stop is CompositeFermat, otherwise zero.
  LucasPrimalityWork work; ///< Actual selected producing operations.
};
/// @brief One real trial-divisor test and its committed exact removals.
/// @par Example
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(7);
/// std::print("{}\n",t.factor_steps.front().multiplicity); // 1
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasFactorStep {
  std::uint64_t divisor; ///< Actual increasing tested candidate, including composite nonfactors.
  std::uint64_t before;  ///< Actual residual before this candidate.
  std::uint64_t after;   ///< Actual residual after all exact removals of this divisor.
  unsigned      multiplicity; ///< Actual removal count, zero for a nonfactor.
};
/// @brief One actual modular-power condition evaluated while searching a witness.
/// @par Example
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(7);
/// std::print("{}\n",t.powers.front().residue); // 1
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasPowerCheck {
  std::uint64_t base; ///< Actual currently tested base.
  std::uint64_t
      factor; ///< Zero denotes Fermat exponent n-minus1; otherwise a distinct prime divisor q.
  std::uint64_t exponent; ///< Actual exponent n-minus1 or (n-minus1)/q.
  std::uint64_t residue;  ///< Actual exact full-width canonical modular-power result.
  bool passed; ///< True if Fermat residue equals1, or factor-condition residue differs from1.
};
/// @brief One actual examined base, after its real short-circuited power checks.
/// @par Example
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(7);
/// std::print("{}\n",t.bases.back().accepted); // true
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasBaseStep {
  std::uint64_t base;          ///< Actual examined base in increasing order.
  bool          fermat_passed; ///< Actual full-exponent congruence result.
  std::uint64_t
       rejected_factor; ///< Actual first q whose proper-order check returned1, otherwise zero.
  bool accepted;        ///< True only if every required proper-order check passed.
};
/// @brief Owned actual factorization and base/power records from one shared producer execution.
/// @par Example
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(7);
/// std::print("{}\n",t.analysis.witness); // 3
/// @endcode
/// @par References
/// - Pratt(1975), Every Prime Has a Succinct Certificate, https://doi.org/10.1137/0204018; exact order evidence here, not a recursive Pratt proof tree.
/// @see lucas_primality_analyze(), lucas_primality_trace()
struct LucasPrimalityTrace {
  LucasPrimalityAnalysis
      analysis; ///< Actual same-run outcome, partial or complete according to stop.
  std::vector<LucasFactorStep> factor_steps; ///< Actual trial/removal records, without refactoring.
  std::vector<LucasPowerCheck>
      powers; ///< Actual canonical modular-power inputs/results, without repowering.
  std::vector<LucasBaseStep> bases; ///< Actual examined base outcomes, preserving short circuits.
  std::size_t captured_entries{0};  ///< One candidate word plus actual factor/power/base records.
};
/// @brief Decide primality by the same bounded exact Lucas producer; throw rather than call exhaustion composite.
/// @param n Full-range unsigned candidate, with trivial small/even filters.
/// @return True exactly for a conclusive prime result; false exactly for conclusive nonprime results.
/// @throws std::length_error On the ordinary one-million trial-divisor or100000-base budget exhaustion.
/// @note Finite complete ordinary boolean results are preserved. Composite Fermat failures now stop immediately,
/// rather than continuing a pointless complete base scan; this does not change the mathematical decision.
/// @code
/// std::print("{}\n",dm::algorithms::lucas_primality_test(1009)); // true
/// @endcode
bool lucas_primality_test(std::uint64_t n);
/// @brief Factor n-minus1 exactly, then seek its least full-order base with explicit independent work budgets.
/// @param n Full-range unsigned candidate.
/// @param max_trial_divisors Actual candidate divisor-test budget, default one million; zero means no trials.
/// @param max_bases Actual base budget, default100000; zero means no base checks.
/// @return Actual conclusive decision or FactorBudget/BaseBudget, never a false composite certificate on exhaustion.
/// @note Division guard d<=remaining/d avoids overflowing d*d. Exact modular products use canonical128-bit arithmetic.
/// Worst-case trial division is O(sqrt(n)); each examined base requests at most1+omega(n-minus1) O(log(n)) powers.
/// This is an exact order criterion, not a scalable replacement for general-purpose64-bit primality tests.
/// @code
/// auto a=dm::algorithms::lucas_primality_analyze(7,100,0);
/// std::print("{}\n",a.conclusive); // false
/// @endcode
LucasPrimalityAnalysis lucas_primality_analyze(
    std::uint64_t n,
    std::size_t   max_trial_divisors = 1000000,
    std::size_t   max_bases          = 100000);
/// @brief Capture actual trial removals, evaluated powers and base outcomes without factoring/search replay.
/// @param n Original full-range unsigned candidate.
/// @param max_trial_divisors Explicit actual trial budget; zero permits no trials.
/// @param max_bases Explicit actual base budget; zero permits no base checks.
/// @param max_entries Optional complete-or-error capture cap, default one million; zero disables only this cap.
/// @return Owned actual history and same-run conclusive or explicitly incomplete outcome.
/// @throws std::length_error Before the next capture record if one candidate word plus actual factor/power/base records exceeds its cap.
/// @note Counts logical records, excluding fields, returned factor copies, capacities, runtime/output copies and bytes.
/// Work-budget exhaustion returns a complete actual bounded history; capture-cap exhaustion throws without exposing a prefix.
/// @code
/// auto t=dm::algorithms::lucas_primality_trace(1001);
/// std::print("{}\n",t.analysis.prime); // false
/// @endcode
LucasPrimalityTrace lucas_primality_trace(
    std::uint64_t n,
    std::size_t   max_trial_divisors = 1000000,
    std::size_t   max_bases          = 100000,
    std::size_t   max_entries        = 1000000);
/// @brief Run actual bounded Lucas studies on every prescribed integer in an inclusive small interval.
/// @param first First candidate.
/// @param last Last candidate, no lower than first and spanning at most10000 integers.
/// @return Actual independently fitted number studies in increasing order; no prime filtering is precomputed.
/// @throws std::invalid_argument For reversed or overly wide intervals.
/// @code
/// auto a=dm::algorithms::lucas_primality_census(3,5);
/// std::print("{}\n",a.size()); // 3
/// @endcode
std::vector<LucasPrimalityAnalysis> lucas_primality_census(std::uint64_t first, std::uint64_t last);
} // namespace datamunge::algorithms
