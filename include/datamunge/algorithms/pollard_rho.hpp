#pragma once
/// @file pollard_rho.hpp
/// @brief Exact uint64 Brent-rho splitting, actual gcd batches and bounded educational polynomial walks.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual producing polynomial-evaluation phase.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_factor_trace(8051);std::print("{}\n",t.history.evaluations[0].phase==dm::algorithms::RhoFactorPhase::Advance); // true
/// @endcode
enum class RhoFactorPhase {
  Advance, ///< Advance y by the current doubling radius before batched comparisons.
  Batch,   ///< Advance y and multiply the actual difference into q.
  Fallback ///< Revisit the saved block one polynomial step at a time after gcd(q,n)=n.
};
/// @brief Producing evaluation and retained-record policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::RhoFactorOptions o;std::print("{}\n",o.max_evaluations); // 10000000
/// @endcode
struct RhoFactorOptions {
  std::size_t max_evaluations{
      10000000}; ///< Actual polynomial calls across the complete split/factorization, excluding primality internals.
  std::size_t max_trace_records{
      1000000}; ///< Actual evaluation/gcd/split records; excludes final factors and working state.
};
/// @brief Selected actual native calls, not timings or a complete instruction count.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_factor_run(8051).work.evaluations>0); // true
/// @endcode
struct RhoFactorWork {
  std::size_t evaluations{0};    ///< Actual g(v)=(v*v+c) mod n calls.
  std::size_t batch_products{0}; ///< Actual q <- q*abs(x-y) mod n products.
  std::size_t gcd_calls{0};      ///< Actual std::gcd calls in normal/fallback blocks.
  std::size_t attempts{0};       ///< Actual polynomial constants selected, starting at one.
  std::size_t radius_doublings{
      0}; ///< Actual end-of-round radius updates, including the final unused update.
  std::size_t primality_calls{0}; ///< Actual canonical deterministic uint64 primality calls.
  std::size_t peeled_twos{0};     ///< Actual even-factor removals in complete factorization.
};
/// @brief One actual polynomial call in its producing split.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_factor_trace(8051).history.evaluations[0].modulus); // 8051
/// @endcode
struct RhoFactorEvaluation {
  std::uint64_t  modulus;  ///< Actual composite being split.
  std::uint64_t  constant; ///< Actual deterministic c.
  std::uint64_t  before;   ///< Actual polynomial input.
  std::uint64_t  after;    ///< Exact full-width modular square-plus-c output.
  RhoFactorPhase phase;    ///< Actual advance, batch or fallback call site.
};
/// @brief Actual gcd boundary and its immediately preceding working values.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_factor_trace(8051).history.gcds[0].constant); // 1
/// @endcode
struct RhoFactorGcd {
  std::uint64_t modulus;  ///< Actual composite being split.
  std::uint64_t constant; ///< Actual polynomial c.
  std::uint64_t x;        ///< Saved comparison point.
  std::uint64_t y;        ///< Actual current batch/fallback cursor.
  std::uint64_t product;  ///< Actual q for normal batch, actual abs(x-y) for fallback.
  std::uint64_t divisor;  ///< Actual gcd(product,modulus).
  std::uint64_t steps;    ///< Actual batch length, or one for fallback.
  bool          fallback; ///< True only for saved-block single-step isolation.
};
/// @brief One actual successful composite split in complete recursive factorization.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::factorize_trace(8051).history.splits[0].composite); // 8051
/// @endcode
struct RhoFactorSplit {
  std::uint64_t composite; ///< Actual recursively split integer.
  std::uint64_t divisor;   ///< Actual verified proper divisor returned by Brent's producer.
};
/// @brief Owned actual evaluation/gcd/split histories, shared by native trace results.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_factor_trace(8051);std::print("{}\n",t.history.evaluations.size()==t.work.evaluations); // true
/// @endcode
struct RhoFactorHistory {
  std::vector<RhoFactorEvaluation>
                            evaluations; ///< Every actual retained producing polynomial call.
  std::vector<RhoFactorGcd> gcds;        ///< Every actual retained normal/fallback gcd.
  std::vector<RhoFactorSplit>
      splits; ///< Actual recursive split decisions; empty for single-split trace.
};
/// @brief Owned single-split divisor and selected work, without historical records.
/// @par Example
/// @code
/// const auto r=dm::algorithms::pollard_rho_factor_run(8051);std::print("{}\n",8051%r.factor); // 0
/// @endcode
struct RhoFactorRun {
  std::uint64_t factor{0}; ///< Proper divisor for composite input, otherwise unchanged n.
  RhoFactorWork work;      ///< Actual selected producing calls.
};
/// @brief Owned complete prime multiset and selected work, without historical records.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::factorize_run(8051).factors); // [83, 97]
/// @endcode
struct FactorizationRun {
  std::vector<std::uint64_t> factors; ///< Actual ascending primes with multiplicity.
  RhoFactorWork              work;    ///< Actual selected calls over the complete factorization.
};
/// @brief Owned actual single-split outcome and producing records.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_factor_trace(8051);std::print("{}\n",t.history.gcds.size()==t.work.gcd_calls); // true
/// @endcode
struct RhoFactorTrace {
  std::uint64_t    factor{0}; ///< Proper divisor for a composite; n for prime or n<2.
  RhoFactorWork    work;      ///< Actual selected producing counts.
  RhoFactorHistory history;   ///< Actual optional histories; no recurrence replay.
};
/// @brief Owned actual complete prime multiset and recursive producing histories.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::factorize_trace(8051).history.splits.size()); // 1
/// @endcode
struct FactorizationTrace {
  std::vector<std::uint64_t> factors; ///< Ascending prime factors with multiplicity; empty for one.
  RhoFactorWork              work; ///< Actual counts accumulated over the complete factorization.
  RhoFactorHistory history; ///< Actual evaluations/gcds/splits across all producing subproblems.
};
/// @brief Split uint64 n using deterministic c=1,2,... Brent-rho batches of128.
/// Modular products and square-plus-c addition are exact across the entire
/// uint64 range. Prime checks use the canonical deterministic twelve-base API.
/// n<2 returns n; even n returns two; odd prime n returns n. Otherwise returns
/// an actual proper divisor. Heuristic n^(1/4) behavior is not a finite work
/// guarantee: the default polynomial-evaluation cap is ten million. Ordinary
/// execution keeps O(1) producer state and no history; inputs are scalar/local.
/// @param n Integer to split, possibly zero/one or prime.
/// @return Proper divisor for a composite, otherwise the documented unchanged n.
/// @throws std::length_error If the default polynomial-evaluation policy is exhausted.
/// @throws std::overflow_error If radius/counter ranges cannot continue safely.
/// @throws std::runtime_error If all native constants fail to expose a proper divisor.
/// @par Example
/// @code
/// const auto d=dm::algorithms::pollard_rho_factor(8051);std::print("{}\n",d>1&&d<8051&&8051%d==0); // true
/// @endcode
/// @par References
/// - R. P. Brent (1980), An improved Monte Carlo factorization algorithm,
///   BIT20,176-184, doi:10.1007/BF01933190.
[[nodiscard]] std::uint64_t pollard_rho_factor(std::uint64_t n);
/// @brief Run the shared single-split producer under configurable policies without history.
/// @param n Integer to split, including prime and small-input conventions.
/// @param options Actual cumulative polynomial-evaluation policy; history cap unused.
/// @return Owned actual divisor and selected work, with no retained trace.
/// @throws std::length_error If evaluation policy is exhausted.
/// @throws std::overflow_error If radius/counter ranges cannot continue.
/// @throws std::runtime_error If all native constants fail.
/// @par Example
/// @code
/// const auto r=dm::algorithms::pollard_rho_factor_run(8051);std::print("{}\n",r.factor>1&&r.factor<8051); // true
/// @endcode
RhoFactorRun pollard_rho_factor_run(std::uint64_t n, RhoFactorOptions options = {});
/// @brief Capture every actual polynomial/gcd call of the shared single-split producer.
/// @param n Integer to split, with the ordinary prime/small-input conventions.
/// @param options Actual evaluation and retained-record policies.
/// @return Owned actual divisor, selected work and producing records.
/// @throws std::length_error If evaluation/history policy is exhausted.
/// @throws std::overflow_error If radius/counter ranges cannot continue.
/// @throws std::runtime_error If native constants fail to expose a proper divisor.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_factor_trace(8051).history.evaluations.empty()); // false
/// @endcode
RhoFactorTrace pollard_rho_factor_trace(std::uint64_t n, RhoFactorOptions options = {});
/// @brief Produce the ascending prime multiset by peeling twos and recursively splitting odd composites.
/// Uses one cumulative default polynomial-evaluation cap over the complete
/// factorization; no optional history is retained. At most64 prime factors and
/// fixed-width recursive depth; no shared mutable state or retained references.
/// @param n Positive integer, including one.
/// @return Owned sorted primes with multiplicity; their exact product is n.
/// @throws std::invalid_argument If n is zero.
/// @throws std::length_error If the default cumulative evaluation policy is exhausted.
/// @throws std::overflow_error If producer radius/counter ranges cannot continue.
/// @throws std::runtime_error If a composite cannot be split by the native constants.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::factorize(8051)); // [83, 97]
/// @endcode
[[nodiscard]] std::vector<std::uint64_t> factorize(std::uint64_t n);
/// @brief Run the shared complete factorization under configurable cumulative policy without history.
/// @param n Positive integer, including one.
/// @param options Actual cumulative evaluation policy; history cap unused.
/// @return Owned ascending prime multiset and selected work, without history.
/// @throws std::invalid_argument If n is zero.
/// @throws std::length_error If cumulative evaluation policy is exhausted.
/// @throws std::overflow_error If producer radius/counter ranges cannot continue.
/// @throws std::runtime_error If a native split fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::factorize_run(8051).factors); // [83, 97]
/// @endcode
FactorizationRun factorize_run(std::uint64_t n, RhoFactorOptions options = {});
/// @brief Capture actual recursive split histories under one complete-factorization policy.
/// @param n Positive integer, including one.
/// @param options Actual cumulative evaluation and retained-record policies.
/// @return Owned prime multiset and actual producing records across subproblems.
/// @throws std::invalid_argument If n is zero.
/// @throws std::length_error If evaluation/history policy is exhausted.
/// @throws std::overflow_error If radius/counter ranges cannot continue.
/// @throws std::runtime_error If a native split cannot expose a proper divisor.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::factorize_trace(8051).factors); // [83, 97]
/// @endcode
FactorizationTrace factorize_trace(std::uint64_t n, RhoFactorOptions options = {});
/// @brief Owned educational polynomial orbit and first repeated projected residue.
/// This is a separate forward-orbit fixture, not Brent's cursor visitation path.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rho_polynomial_walk(8051,1,2,16,83).period); // 5
/// @endcode
struct RhoPolynomialWalk {
  std::vector<std::uint64_t>
      states; ///< Actual complete-modulus states, including the normalized starting state.
  std::vector<std::uint64_t>
              projected;       ///< Actual state modulo the supplied divisor, from the same visits.
  bool        repeated{false}; ///< True if a projected residue repeated within the retained prefix.
  std::size_t cycle_head{
      0}; ///< First occurrence of the first repeated projected residue, meaningful only when repeated.
  std::size_t period{0}; ///< Actual first-repeat index minus cycle_head.
};
/// @brief Run the shared exact polynomial as a separately labelled forward-orbit fixture.
/// Detects the first repeated projected residue from actual visits while keeping
/// the complete requested prefix. A nonzero projection must divide n; zero uses n.
/// No factor search or Brent schedule is simulated here. O(count) time/storage.
/// @param n Modulus >=2.
/// @param c Polynomial constant, reduced modulo n.
/// @param start Starting state, reduced modulo n.
/// @param count Number of states, possibly zero; count-1 actual transitions when nonempty.
/// @param projection Zero for n, otherwise a divisor >=2 of n.
/// @param max_entries Retained states plus projected values, default one million; zero disables policy.
/// @return Owned actual forward states and first-repeat diagnostics; no replay.
/// @throws std::invalid_argument If modulus or projection is invalid.
/// @throws std::length_error If count/storage/history ranges cannot be admitted.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rho_polynomial_walk(8051,1,2,16,83).period); // 5
/// @endcode
RhoPolynomialWalk rho_polynomial_walk(
    std::uint64_t n,
    std::uint64_t c,
    std::uint64_t start,
    std::size_t   count,
    std::uint64_t projection  = 0,
    std::size_t   max_entries = 1000000);
} // namespace datamunge::algorithms
