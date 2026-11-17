#pragma once
/// @file pollard_p_minus_one.hpp
/// @brief Actual base-two factorial-exponent p-minus-one attempts and gcd decisions.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Exact native stopping reason, not a primality or factor-existence certificate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_one_run(85,4).stop==dm::algorithms::PMinusOneStop::Collapsed); // true
/// @endcode
enum class PMinusOneStop {
  BoundExhausted, ///< Every j from 2 through B completed without a proper factor or full collapse.
  EvenFactor,     ///< Even n>2 returns divisor two without exponentiation.
  ProperFactor,   ///< Actual gcd lies strictly between one and n.
  Collapsed,      ///< Actual gcd equals n, so this fixed-base attempt stops without a divisor.
  SmallPrime      ///< Input two has no nontrivial factor and returns divisor zero.
};
/// @brief Selected factorial-stage and retained-record policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::PMinusOneOptions o;std::print("{}\n",o.max_stages); // 1000000
/// @endcode
struct PMinusOneOptions {
  std::size_t max_stages{
      1000000}; ///< Actual powering/gcd stage cap; excludes powering's internal bit operations.
  std::size_t max_trace_records{1000000}; ///< Actual stage-record cap; excludes ordinary result.
};
/// @brief One actual a <- a^j mod n update and immediately tested gcd.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_one_trace(391,4).stages[0].after); // 4
/// @endcode
struct PMinusOneStage {
  std::uint64_t j;      ///< Actual factorial multiplier j, beginning at two.
  std::uint64_t before; ///< Residue before this powering call.
  std::uint64_t after;  ///< Actual pow(before,j,n), representing 2^(j!) mod n.
  std::uint64_t
      gcd; ///< Actual gcd(after-1,n), with residue zero handled without unsigned underflow.
};
/// @brief Owned actual result and selected call-boundary work.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_one_run(391,4).divisor); // 17
/// @endcode
struct PMinusOneRun {
  std::uint64_t divisor{0}; ///< Proper factor or zero; even n>2 gives two, n=2 gives zero.
  PMinusOneStop stop{PMinusOneStop::BoundExhausted}; ///< Actual stopping reason.
  std::size_t   stages{0}; ///< Actual powering/gcd pairs performed, not a theoretical bound.
};
/// @brief Owned ordinary outcome and every actual producing factorial stage.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_one_trace(391,4).stages.back().gcd); // 17
/// @endcode
struct PMinusOneTrace {
  PMinusOneRun                run;    ///< Ordinary output and actual selected count.
  std::vector<PMinusOneStage> stages; ///< Actual residue/gcd records, no replay.
};
/// @brief Attempt a proper divisor with the native base-two factorial-exponent schedule.
/// Starts a=2 mod n, then a=pow(a,j,n) and gcd(a-1,n) for j=2,...,B.
/// Thus exponents are j!, not the usual lcm(1,...,B) prime-power stage-one
/// construction. Base order dividing j! is sufficient for divisibility; smooth
/// p-1 alone or reaching its largest prime does not ensure an isolating gcd.
/// Stops at the first proper gcd or full n collapse. Failure is not primality
/// or absence-of-factor evidence. Input is scalar, no shared mutable state.
/// @param n Integer >=2; two returns zero, even n>2 returns two immediately.
/// @param bound Inclusive maximum j, possibly zero/one for no stages.
/// @return Proper divisor or zero using default one-million-stage policy.
/// @throws std::invalid_argument If n<2.
/// @throws std::length_error If default stage policy is exhausted before a result.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_1(17*23,4)); // 17
/// @endcode
/// @par References
/// - J. M. Pollard (1974), Theorems on factorization and primality testing,
///   Proceedings of the Cambridge Philosophical Society 76, 521-528,
///   doi:10.1017/S0305004100049252. The factorial schedule is the native variant.
std::uint64_t pollard_p_minus_1(std::uint64_t n, std::uint64_t bound);
/// @brief Run the shared factorial-exponent producer with actual count and reason.
/// Work is the sum of O(log j) modular powering steps plus gcd work over executed
/// stages; O(1) producer state beyond fixed-width arithmetic. No history is kept.
/// @param n Integer >=2.
/// @param bound Inclusive maximum factorial multiplier, possibly below two.
/// @param options Actual-stage/history policies; history policy unused here.
/// @return Owned divisor, actual stop and actual stage count.
/// @throws std::invalid_argument If n<2.
/// @throws std::length_error If selected stage policy is exhausted.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_one_run(391,3).divisor); // 0
/// @endcode
PMinusOneRun
pollard_p_minus_one_run(std::uint64_t n, std::uint64_t bound, PMinusOneOptions options = {});
/// @brief Capture each actual residue update and gcd from the shared attempt.
/// Retains one fixed record per executed stage, not a reconstructed factorial.
/// @param n Integer >=2.
/// @param bound Inclusive maximum j.
/// @param options Actual-stage and retained-record policies, checked before each producing call.
/// @return Owned ordinary outcome and actual complete stage records.
/// @throws std::invalid_argument If n<2.
/// @throws std::length_error If selected stage/history policy is exhausted.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_p_minus_one_trace(391,4).stages.size()); // 3
/// @endcode
PMinusOneTrace
pollard_p_minus_one_trace(std::uint64_t n, std::uint64_t bound, PMinusOneOptions options = {});
} // namespace datamunge::algorithms
