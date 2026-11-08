#pragma once
/// @file chinese_remainder.hpp
/// @brief Checked generalized CRT merging, actual Bezout certificates and explicit period-range diagnostics.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned representable generalized CRT result, retaining the legacy field order and defaults.
/// A successful solve has 0<=remainder<modulus, with modulus the lcm of the
/// processed input moduli. Inconsistency returns {false,0,0}; the default record
/// {false,0,1} is merely unverified storage. Empty input solves to {true,0,1}.
/// Period overflow throws in chinese_remainder(), rather than masquerading as
/// mathematical inconsistency. Manually constructed records are not validated.
/// @par Example
/// @code
/// const auto s=dm::algorithms::chinese_remainder({2,3,2},{3,5,7});
/// std::print("{} {} {}\n",s.solvable,s.remainder,s.modulus); // true 23 105
/// @endcode
/// @see chinese_remainder(), CrtTrace
struct CrtSolution {
  bool solvable{false}; ///< True for a completed representable solution; false for inconsistency or unverified default storage.
  std::int64_t remainder{0}; ///< Canonical residue on success; zero for the inconsistent sentinel.
  std::int64_t modulus{1}; ///< Positive lcm period on success; zero for the inconsistent sentinel.
};
/// @brief Actual outcome of a whole CRT run or one attempted merge.
/// Overflow means a compatible prefix's period exceeds INT64_MAX, without a
/// mathematical verdict on unprocessed constraints. Complete means a compatible,
/// representable merge or a fully solved system, according to its record context.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chinese_remainder_trace({1,0},{2,4});
/// std::print("{}\n",t.status==dm::algorithms::CrtStatus::Inconsistent); // true
/// @endcode
/// @see CrtTrace, CrtMerge
enum class CrtStatus {
  Unprocessed, ///< Default storage; no attempted solve or merge is represented.
  Complete, ///< All inputs in the record's scope are compatible and their period fits int64.
  Inconsistent, ///< The actual gcd divisibility test failed.
  Overflow ///< A compatible merge has a period above INT64_MAX; no int64 pair is returned.
};
/// @brief Named complete-call limits for attempted merges and actual Euclidean divisions.
/// Zero means unlimited. Limits count operations until the first inconsistency or
/// period overflow, not the unvisited suffix. All moduli are validated before
/// these limits apply. No partial result/trace escapes when a work cap is reached.
/// @par Example
/// @code
/// dm::algorithms::CrtOptions options;options.max_merges=3;options.max_euclidean_steps=8;
/// const auto t=dm::algorithms::chinese_remainder_trace({2,3,2},{3,5,7},options);
/// std::print("{}\n",t.work.euclidean_steps); // 8
/// @endcode
/// @see chinese_remainder_trace(), chinese_remainder_with_options()
struct CrtOptions {
  std::size_t max_merges{0}; ///< Maximum attempted merges; zero is unlimited.
  std::size_t max_euclidean_steps{0}; ///< Maximum division iterations across all actual extended-gcd calls; zero is unlimited.
};
/// @brief Actual validation visits, attempted merges and named arithmetic operations.
/// Euclidean steps include an initial zero quotient when the first modulus is
/// smaller. Products count the correction product, period product and residue
/// product separately, including compatible overflow attempts and redundant/unit
/// congruences. Validation covers every modulus even if arithmetic stops early.
/// These are fixed-word counts, not elapsed time or arbitrary-precision bit cost.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chinese_remainder_trace({2,3,2},{3,5,7});
/// std::print("{} {} {}\n",t.work.validated_moduli,t.work.attempted_merges,t.work.completed_merges); // 3 3 3
/// @endcode
/// @see CrtOptions, CrtTrace
struct CrtWork {
  std::size_t validated_moduli{0}; ///< Positive input moduli checked before arithmetic.
  std::size_t attempted_merges{0}; ///< Actual gcd/compatibility attempts, including a failing final one.
  std::size_t completed_merges{0}; ///< Compatible merges whose period fit and were installed as the next prefix.
  std::size_t euclidean_steps{0}; ///< Actual quotient/remainder iterations across all merges.
  std::size_t correction_products{0}; ///< Actual reduced-difference times reduced-Bezout-coefficient products.
  std::size_t period_products{0}; ///< Actual prior-period times reduced-input-modulus products.
  std::size_t residue_products{0}; ///< Actual prior-period times normalized-multiplier products.
};
/// @brief Owned actual certificate and wide candidate for one sequential CRT merge.
/// before is the last representable compatible prefix. Bezout coefficients obey
/// before.modulus*bezout_first+input_modulus*bezout_second=gcd. difference is the
/// normalized input residue minus before.remainder. Inconsistency leaves the
/// correction/candidate fields zero and emits no hypothetical merged pair.
/// Compatible attempts retain the actual signed correction product, normalized
/// multiplier and wide candidate, even when the period cannot fit int64.
/// @par Example
/// @code
/// const auto s=dm::algorithms::chinese_remainder_trace({2,3,2},{3,5,7}).steps[1];
/// std::print("{} {} {}\n",s.gcd,s.multiplier,dm::algorithms::int128_text(s.candidate_remainder)); // 1 2 8
/// @endcode
/// @see CrtTrace, CrtStatus
struct CrtMerge {
  std::size_t index{0}; ///< Original input position, zero based.
  CrtSolution before; ///< Actual representable prefix before this attempt.
  std::int64_t input_remainder{0}; ///< Original signed residue, before normalization.
  std::int64_t input_modulus{1}; ///< Actual positive input modulus.
  std::int64_t normalized_remainder{0}; ///< Input residue reduced into [0,input_modulus).
  std::int64_t gcd{0}; ///< Actual positive gcd of the two moduli.
  __int128 bezout_first{0}; ///< Actual coefficient of before.modulus in the gcd identity.
  __int128 bezout_second{0}; ///< Actual coefficient of input_modulus in the gcd identity.
  std::int64_t difference{0}; ///< normalized_remainder-before.remainder, safely within int64.
  std::int64_t reduced_modulus{1}; ///< input_modulus/gcd; one is permitted.
  __int128 correction_product{0}; ///< Actual ((difference/gcd)%reduced_modulus)*(bezout_first%reduced_modulus), only for compatible attempts.
  std::int64_t multiplier{0}; ///< Correction product normalized modulo reduced_modulus, zero for inconsistent attempts.
  __int128 candidate_remainder{0}; ///< Actual canonical wide residue, zero when inconsistent.
  __int128 candidate_modulus{0}; ///< Actual wide lcm period, zero when inconsistent.
  CrtStatus status{CrtStatus::Unprocessed}; ///< Actual compatible, inconsistent or period-overflow outcome.
  std::size_t euclidean_steps{0}; ///< Actual division iterations in this attempt's gcd computation.
};
/// @brief Owned actual sequential CRT path with an optional complete int64 verdict.
/// Complete and Inconsistent runs fill solution with their legacy result;
/// Overflow leaves solution empty. prefix always retains the last representable
/// compatible prefix, including on failure. steps includes the first rejected
/// attempt and omits the unvisited suffix. Inconsistency/overflow are actual
/// mathematical/range outcomes; work caps and allocation failures still throw,
/// exposing no partial trace. Default construction alone performs no solve.
/// Ordinary and traced entry points share the executing kernel without replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chinese_remainder_trace({2,3,2},{3,5,7});
/// std::print("{} {}\n",t.steps.size(),t.solution->remainder); // 3 23
/// @endcode
/// @see chinese_remainder_trace(), CrtMerge, CrtWork
struct CrtTrace {
  CrtStatus status{CrtStatus::Unprocessed}; ///< Whole-call arithmetic outcome, distinct from storage defaults.
  std::optional<CrtSolution> solution; ///< Complete representable verdict, absent only for overflow or default storage.
  CrtSolution prefix{true,0,1}; ///< Last actually installed compatible representable congruence.
  std::vector<CrtMerge> steps; ///< Actual merge records, including the first failing attempt.
  CrtWork work; ///< Actual validated and executed operation counts.
};
/// @brief Normalize a signed residue into [0,modulus) without overflowing value+modulus.
/// Handles INT64_MIN and every positive int64 modulus, including one. No
/// allocation or mutation; constant fixed-word work and no shared state.
/// @param value Arbitrary signed residue.
/// @param modulus Strictly positive signed modulus.
/// @return Unique nonnegative representative smaller than modulus.
/// @throws std::invalid_argument If modulus is nonpositive.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::crt_normalize(-3,7)); // 4
/// @endcode
/// @see crt_residue_satisfies(), chinese_remainder()
std::int64_t crt_normalize(std::int64_t value,std::int64_t modulus);
/// @brief Check whether one signed integer satisfies every supplied congruence.
/// Validates vector lengths and all positive moduli before comparing values,
/// even if an earlier comparison would fail. Empty inputs return true. This
/// checks a representative, not a claimed solution period or a uniqueness proof.
/// No input mutation/allocation; O(n) modular comparisons, constant storage.
/// @param candidate Arbitrary signed candidate integer, including INT64_MIN.
/// @param remainders Borrowed signed target residues.
/// @param moduli Borrowed strictly positive moduli, matching remainders length.
/// @return True iff candidate is congruent to every target residue.
/// @throws std::invalid_argument If lengths differ or any modulus is nonpositive.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::crt_residue_satisfies(23,{2,3,2},{3,5,7})); // true
/// @endcode
/// @see crt_normalize(), chinese_remainder()
bool crt_residue_satisfies(std::int64_t candidate,const std::vector<std::int64_t>& remainders,
                          const std::vector<std::int64_t>& moduli);
/// @brief Solve generalized CRT by checked sequential merging, permitting shared factors.
/// Keeps the legacy call and CrtSolution layout. Equal-length inputs and every
/// positive modulus are validated before merging. Residues may be any int64.
/// Starts at 0 modulo 1; empty inputs return {true,0,1}. A failed gcd divisibility
/// test returns {false,0,0}. Compatible merges use signed-128 products/Bezout
/// arithmetic and check the new lcm before narrowing. A consistent prefix whose
/// period exceeds INT64_MAX throws; later constraints are not examined, so this
/// is not a verdict on that suffix's mathematical consistency. Duplicate and unit
/// moduli are supported. Ordinary calls allocate no provenance and mutate no input.
/// O(n) validation plus Euclidean division work per attempted merge; constant
/// auxiliary storage, not an arbitrary-precision bit-complexity claim.
/// @param remainders Borrowed signed target residues, normalized per attempted merge.
/// @param moduli Borrowed positive moduli, same length as remainders.
/// @return Canonical representable solution and lcm, or inconsistent sentinel.
/// @throws std::invalid_argument If lengths differ or any modulus is nonpositive.
/// @throws std::overflow_error If a compatible prefix period exceeds int64 or a work counter overflows.
/// @par Example
/// @code
/// const auto s=dm::algorithms::chinese_remainder({2,8},{6,12});
/// std::print("{} {}\n",s.remainder,s.modulus); // 8 12
/// @endcode
/// @par References
/// - V. Shoup (2008), A Computational Introduction to Number Theory and Algebra,
///   Sections 2.3 and 4.3, https://www.shoup.net/ntb/ntb-v2_5.pdf.
///   Pairwise generalized merging applies the gcd linear-congruence criterion.
/// @see chinese_remainder_trace(), chinese_remainder_with_options(), crt_residue_satisfies()
CrtSolution chinese_remainder(const std::vector<std::int64_t>& remainders,
                              const std::vector<std::int64_t>& moduli);
/// @brief Solve with the same checked kernel and explicit attempted-merge/division caps.
/// Same result, validation, range and sequential-stop semantics as
/// chinese_remainder(). Positive caps count actual operations; zero is unlimited.
/// Inconsistency before a cap returns its verdict even if more inputs remain.
/// Cap failures expose no partial result; ordinary calls retain no provenance.
/// @param remainders Borrowed signed target residues.
/// @param moduli Borrowed positive moduli, matching remainders length.
/// @param options Complete-call work limits, default unlimited.
/// @return Completed legacy verdict, without a partial prefix result on overflow/cap failure.
/// @throws std::invalid_argument If lengths differ or any modulus is nonpositive.
/// @throws std::overflow_error If a compatible prefix period exceeds int64 or a counter overflows.
/// @throws std::length_error If an actual merge/division cap is exhausted.
/// @par Example
/// @code
/// dm::algorithms::CrtOptions options;options.max_merges=1;
/// const auto s=dm::algorithms::chinese_remainder_with_options({-3},{7},options);
/// std::print("{}\n",s.remainder); // 4
/// @endcode
/// @par References
/// - V. Shoup (2008), https://www.shoup.net/ntb/ntb-v2_5.pdf, Sections 2.3 and 4.3.
/// @see chinese_remainder(), chinese_remainder_trace(), CrtOptions
CrtSolution chinese_remainder_with_options(const std::vector<std::int64_t>& remainders,
    const std::vector<std::int64_t>& moduli,CrtOptions options={});
/// @brief Retain actual sequential merges, Bezout certificates and wide range-failure diagnostics.
/// Uses the same executing kernel, without replay. A compatible wide period is
/// retained as status=Overflow with solution empty and prefix unchanged; unlike
/// the ordinary solver this range outcome does not throw. Inconsistency fills
/// solution with {false,0,0}. Stops at the first arithmetic outcome and records
/// no suffix merges, although all moduli were validated first. Budget/allocation
/// failures still throw and return no partial trace. O(k) owned records for k
/// attempted merges, plus constant numeric state; no caller input is mutated.
/// @param remainders Borrowed signed target residues.
/// @param moduli Borrowed positive moduli, matching remainders length.
/// @param options Complete-call merge/division limits, default unlimited.
/// @return Owned actual path, last representable prefix, optional verdict and counters.
/// @throws std::invalid_argument If lengths differ or any modulus is nonpositive.
/// @throws std::overflow_error If an actual work counter overflows size_t.
/// @throws std::length_error If a work cap or vector record capacity is exceeded.
/// @throws std::bad_alloc If trace allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chinese_remainder_trace({1,0},{2,4});
/// std::print("{} {}\n",t.steps.back().gcd,t.solution->solvable); // 2 false
/// @endcode
/// @par References
/// - V. Shoup (2008), https://www.shoup.net/ntb/ntb-v2_5.pdf, Sections 2.3 and 4.3.
/// @see chinese_remainder(), CrtTrace, CrtStatus
CrtTrace chinese_remainder_trace(const std::vector<std::int64_t>& remainders,
    const std::vector<std::int64_t>& moduli,CrtOptions options={});
} // namespace datamunge::algorithms
