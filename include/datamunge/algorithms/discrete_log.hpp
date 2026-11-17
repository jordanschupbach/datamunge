#pragma once
/// @file discrete_log.hpp
/// @brief Checked finite-field rho and native linear-digit Pohlig-Hellman logarithms.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Selected primitive-call and actual-history policies.
/// @par Example
/// @code
/// const dm::algorithms::DiscreteLogOptions o;std::print("{}\n",o.max_attempts); // 40
/// @endcode
struct DiscreteLogOptions {
  std::size_t max_operations{
      1000000}; ///< Selected primality/power/inverse/product calls, trial divisions and trial comparisons; zero disables, excludes primitive internals.
  std::size_t max_trace_records{
      1000000}; ///< Actual step/collision/digit/CRT records; zero disables, excludes fixed result/factor metadata.
  std::size_t max_attempts{
      40}; ///< Rho restarts; zero returns unresolved unless target is identity. Unused by Pohlig-Hellman.
};
/// @brief Actual selected primitive-call counts; these are not timings or all bit operations.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_run(2,8,13,12).work.small_log_tests>0); // true
/// @endcode
struct DiscreteLogWork {
  std::size_t operations{0};      ///< Sum of admitted selected primitive calls and trial checks.
  std::size_t primality_calls{0}; ///< Calls to the canonical deterministic uint64 primality test.
  std::size_t power_calls{
      0}; ///< Calls to canonical mod_pow; its internal bit operations are excluded.
  std::size_t inverse_calls{
      0}; ///< Calls to canonical signed64 modular_inverse on validated positive moduli.
  std::size_t modular_products{
      0}; ///< Actual mul_mod calls outside mod_pow/primality/inverse internals.
  std::size_t factor_trials{0}; ///< Actual trial-divisor checks while factoring the declared order.
  std::size_t small_log_tests{0}; ///< Actual candidate comparisons in linear subgroup digit search.
  std::size_t attempts{0};        ///< Actual rho starting states selected.
  std::size_t walk_updates{
      0}; ///< Actual single-cursor group updates; three per completed Floyd iteration.
  std::size_t collisions{0};      ///< Actual tortoise/hare equal-value comparisons that succeeded.
  std::size_t candidate_tests{0}; ///< Actual verified full logarithm candidates after a collision.
};
/// @brief Owned actual exponent outcome and selected work.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_run(2,8,13,12).exponent); // 3
/// @endcode
struct DiscreteLogRun {
  long long exponent{-1}; ///< Verified solution in [0,order), or -1 when unresolved.
  bool found{false}; ///< True only after an actual equation check, or validated identity shortcut.
  DiscreteLogWork work; ///< Actual selected counts from this run.
};
/// @brief One multiplicative-walk invariant state, value = g^a*h^b mod p.
/// @par Example
/// @code
/// const dm::algorithms::RhoLogState s{1,0,0};std::print("{}\n",s.value); // 1
/// @endcode
struct RhoLogState {
  std::uint64_t value{0}; ///< Actual current group element.
  std::uint64_t a{0};     ///< Actual exponent of g modulo declared order.
  std::uint64_t b{0};     ///< Actual exponent of h modulo declared order.
};
/// @brief One actual branch and cursor update in a rho walk.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_log_trace(2,8,1019,1018).steps[0].hare); // false
/// @endcode
struct RhoLogStep {
  std::size_t attempt;   ///< Zero-based restart id, zero for a standalone educational walk.
  std::size_t iteration; ///< One-based completed Floyd-iteration candidate.
  bool        hare;      ///< False for tortoise; true for either of the two hare updates.
  unsigned branch; ///< Actual old value modulo three, selecting multiply-h, square or multiply-g.
  RhoLogState before; ///< Actual pre-update invariant state.
  RhoLogState after;  ///< Actual post-update invariant state.
};
/// @brief Actual collision congruence and its candidate-verification outcome.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_log_trace(2,8,1019,1018);std::print("{}\n",t.collisions[0].tortoise.value==t.collisions[0].hare.value); // true
/// @endcode
struct RhoLogCollision {
  std::size_t   attempt;            ///< Actual restart id.
  std::size_t   iteration;          ///< Actual one-based Floyd iteration where values agree.
  RhoLogState   tortoise;           ///< Actual collision state.
  RhoLogState   hare;               ///< Actual other collision state.
  std::uint64_t numerator{0};       ///< (a-A) modulo declared order.
  std::uint64_t denominator{0};     ///< (B-b) modulo declared order.
  std::uint64_t gcd{0};             ///< gcd(denominator,order), zero for denominator-zero restart.
  std::uint64_t reduced_modulus{0}; ///< order/gcd for a compatible nondegenerate congruence.
  bool          compatible{false};  ///< True when gcd divides the numerator.
  bool          resolved{false};    ///< True when an enumerated candidate was actually verified.
};
/// @brief Owned ordinary rho result and actual producing walk/collision records.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_log_trace(2,8,1019,1018);std::print("{}\n",t.steps.size()==t.run.work.walk_updates); // true
/// @endcode
struct RhoLogTrace {
  DiscreteLogRun               run;   ///< Actual solver result and work.
  std::vector<RhoLogStep>      steps; ///< Every actual cursor update in the retained producing run.
  std::vector<RhoLogCollision> collisions; ///< Actual congruence decisions; no separate replay.
};
/// @brief Actual prime-power factor selected by the Pohlig-Hellman producer.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_trace(2,8,13,12).factors[0].power); // 4
/// @endcode
struct PohligFactor {
  std::uint64_t prime;    ///< Prime divisor q of the declared exact base order.
  unsigned      exponent; ///< Multiplicity e in the order.
  std::uint64_t power;    ///< Actual q^e, at most declared order.
};
/// @brief Actual linear-search digit solve within one prime-power factor.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_trace(2,8,13,12).digits[0].digit); // 1
/// @endcode
struct PohligDigit {
  std::uint64_t prime;        ///< Current subgroup prime q.
  unsigned      index;        ///< Zero-based base-q digit index.
  std::uint64_t prefix_power; ///< q^index before adding this digit.
  std::uint64_t before;       ///< Previously solved residue modulo prefix_power.
  std::uint64_t target;       ///< Actual reduced target passed to the native linear digit search.
  long long     digit;        ///< Actual selected digit, or -1 if that search failed.
  std::size_t   candidate_tests; ///< Actual candidate equality comparisons for this digit.
  std::uint64_t after;           ///< Actual updated residue, or unchanged before on failure.
};
/// @brief Actual incremental Chinese-remainder merge of one solved factor.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_trace(2,8,13,12).merges[0].combined_modulus); // 4
/// @endcode
struct PohligMerge {
  std::uint64_t modulus;          ///< Current prime-power modulus q^e.
  std::uint64_t residue;          ///< Actual solved residue modulo this factor.
  std::uint64_t previous_value;   ///< CRT value before this factor.
  std::uint64_t previous_modulus; ///< Product of earlier pairwise-coprime factors.
  std::uint64_t after;            ///< Actual merged nonnegative representative.
  std::uint64_t combined_modulus; ///< Product after the merge, at most declared order.
};
/// @brief Owned actual native factor/digit/CRT histories.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_trace(2,8,13,12).run.exponent); // 3
/// @endcode
struct PohligHellmanTrace {
  DiscreteLogRun run; ///< Verified ordinary result and actual selected work.
  std::vector<PohligFactor>
      factors;                     ///< Actual trial-division order decomposition, ascending primes.
  std::vector<PohligDigit> digits; ///< Actual linear-search digit outcomes.
  std::vector<PohligMerge> merges; ///< Actual incremental CRT states.
};
/// @brief Solve g^x=h in a prime field using the native deterministic-seeded rho walk.
/// Reduces g,h modulo p; requires both nonzero, prime p, and declared order in
/// [1,LLONG_MAX] dividing p-1 with g^order=h^order=1. The order may be a multiple
/// of the exact base order; candidates are individually verified. A collision
/// implies an exponent congruence modulo the exact ord(g). Solving that
/// congruence modulo a larger declared annihilator is a stronger condition
/// and can discard useful collisions. Uses the
/// historical seed, three-way branch and up to40 restarts, with a default
/// selected-operation cap. An unresolved return is not proof no logarithm exists.
/// O(1) producer storage; restart iteration bounds are checked without wrap.
/// Heuristic square-root collision behavior is not a per-run work guarantee.
/// @param g Unsigned base, reduced modulo p.
/// @param h Unsigned target, reduced modulo p.
/// @param p Prime modulus, including two.
/// @param order Positive annihilating subgroup order representable by long long.
/// @return Verified exponent in [0,order), or -1 when the native attempts fail.
/// @throws std::invalid_argument If modulus, group membership or declared order is invalid.
/// @throws std::length_error If the default selected-operation policy is exhausted.
/// @throws std::overflow_error If checked inverse/CRT arithmetic reports a range failure.
/// @throws std::runtime_error If a checked internal inverse invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_log(2,1,23,11)); // 0
/// @endcode
/// @par References
/// - J. M. Pollard (1978), Monte Carlo methods for index computation (mod p),
///   Mathematics of Computation32,918-924, doi:10.1090/S0025-5718-1978-0491431-9.
long long pollard_rho_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order);
/// @brief Run the shared rho solver with actual work and configurable selected policies.
/// @param g Base reduced modulo p.
/// @param h Target reduced modulo p.
/// @param p Prime modulus.
/// @param order Valid annihilating order in [1,LLONG_MAX].
/// @param options Selected-operation/history policies and finite restart count.
/// @return Owned verified exponent/found flag and actual work; no history retained.
/// @throws std::invalid_argument If group inputs are invalid.
/// @throws std::length_error If selected operation policy is exhausted.
/// @throws std::overflow_error If checked inverse/CRT arithmetic reports a range failure.
/// @throws std::runtime_error If a checked internal inverse invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_log_run(2,1,23,11).found); // true
/// @endcode
DiscreteLogRun pollard_rho_log_run(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions options = {});
/// @brief Capture each actual cursor update and collision from the shared rho solver.
/// @param g Base reduced modulo p.
/// @param h Target reduced modulo p.
/// @param p Prime modulus.
/// @param order Valid annihilating order.
/// @param options Actual operation/history policies and restart count.
/// @return Owned solver run and actual producing records; no reconstruction.
/// @throws std::invalid_argument If group inputs are invalid.
/// @throws std::length_error If selected operation/history policy is exhausted.
/// @throws std::overflow_error If checked inverse/CRT arithmetic reports a range failure.
/// @throws std::runtime_error If a checked internal inverse invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_log_trace(2,8,1019,1018).run.found); // true
/// @endcode
RhoLogTrace pollard_rho_log_trace(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions options = {});
/// @brief Solve by prime-power digits and CRT using the native linear digit search.
/// Requires prime p and exact base order in [1,LLONG_MAX], dividing p-1;
/// validates all prime factors by g^(order/q)!=1. Target must lie in that subgroup.
/// Trial factoring can take O(sqrt(order)) checks; each native digit search scans
/// q candidates linearly, not sqrt(q) and not a prebuilt lookup table. Selected
/// work is capped by default. Fixed-width products and CRT merges are widened;
/// modular inverse delegates to the canonical checked extended-Euclidean API.
/// @param g Nonzero base modulo p with exact declared order.
/// @param h Target in the generated subgroup, reduced modulo p.
/// @param p Prime modulus.
/// @param order Exact base order, positive and representable by long long.
/// @return Verified logarithm in [0,order), or -1 if a digit/equation check fails.
/// @throws std::invalid_argument If group inputs or exact base order are invalid.
/// @throws std::length_error If default selected-operation policy is exhausted.
/// @throws std::overflow_error If checked inverse/CRT arithmetic reports a range failure.
/// @throws std::runtime_error If a checked internal inverse invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_log(2,8,13,12)); // 3
/// @endcode
/// @par References
/// - S. C. Pohlig and M. E. Hellman (1978), An improved algorithm for computing
///   logarithms over GF(p) and its cryptographic significance, doi:10.1109/TIT.1978.1055817.
long long
pohlig_hellman_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order);
/// @brief Run the shared linear-digit Pohlig-Hellman producer with actual selected counts.
/// @param g Base of exact declared order.
/// @param h Subgroup target, reduced modulo p.
/// @param p Prime modulus.
/// @param order Exact positive base order <=LLONG_MAX.
/// @param options Selected operation/history policies; restart count unused.
/// @return Owned verified result and actual counts, without historical records.
/// @throws std::invalid_argument If group or exact-order inputs are invalid.
/// @throws std::length_error If selected operation policy is exhausted.
/// @throws std::overflow_error If checked inverse/CRT arithmetic reports a range failure.
/// @throws std::runtime_error If a checked internal inverse invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_run(2,8,13,12).exponent); // 3
/// @endcode
DiscreteLogRun pohlig_hellman_run(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions options = {});
/// @brief Capture the actual factor decomposition, digit searches and CRT merges.
/// @param g Base of exact declared order.
/// @param h Subgroup target, reduced modulo p.
/// @param p Prime modulus.
/// @param order Exact positive base order <=LLONG_MAX.
/// @param options Selected operation/history policies.
/// @return Owned actual producer histories and verified result, without replay.
/// @throws std::invalid_argument If group or exact-order inputs are invalid.
/// @throws std::length_error If selected operation/history policy is exhausted.
/// @throws std::overflow_error If checked inverse/CRT arithmetic reports a range failure.
/// @throws std::runtime_error If a checked internal inverse invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pohlig_hellman_trace(2,8,13,12).merges.size()); // 2
/// @endcode
PohligHellmanTrace pohlig_hellman_trace(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions options = {});
/// @brief Actual first-collision educational walk, distinct from a complete restarted solver.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_log_first_collision(2,8,1019,1018,1,1);std::print("{}\n",t.steps.size()==3*t.iterations); // true
/// @endcode
struct RhoLogCollisionWalk {
  bool collided{
      false}; ///< Actual equality of cursor group elements within the supplied iteration cap.
  std::size_t iterations{
      0}; ///< Actual completed Floyd iterations, including the collision iteration.
  RhoLogState             tortoise; ///< Actual final slow cursor.
  RhoLogState             hare;     ///< Actual final fast cursor.
  DiscreteLogWork         work;     ///< Selected actual calls and cursor updates.
  std::vector<RhoLogStep> steps; ///< Every actual producing cursor update in this standalone walk.
};
/// @brief Run the shared native walk from explicit exponent state until first collision/cap.
/// This stops before congruence solving; a collision alone is not a logarithm.
/// @param g Nonzero base modulo prime p.
/// @param h Subgroup target modulo p.
/// @param p Prime modulus.
/// @param order Positive annihilating order <=LLONG_MAX.
/// @param a Initial exponent of g, reduced modulo order.
/// @param b Initial exponent of h, reduced modulo order.
/// @param max_iterations Maximum completed Floyd iterations; zero retains only the initial state.
/// @param options Selected operation/history policies; restart count unused.
/// @return Owned actual first-collision/capped states and all producing updates.
/// @throws std::invalid_argument If group inputs are invalid.
/// @throws std::length_error If selected operation/history policy is exhausted.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_rho_log_first_collision(2,8,1019,1018,1,1);std::print("{}\n",t.collided); // true
/// @endcode
RhoLogCollisionWalk pollard_rho_log_first_collision(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    std::uint64_t      a,
    std::uint64_t      b,
    std::size_t        max_iterations = 100000,
    DiscreteLogOptions options        = {});
/// @brief Actual results for one original six-prime, forty-start educational collision setting.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_log_collision_study()[0].trials); // 40
/// @endcode
struct RhoLogCollisionStudy {
  std::uint64_t prime; ///< Original field modulus.
  std::uint64_t base;  ///< Original supplied base.
  std::uint64_t
      declared_order;   ///< Original p-1 annihilating order, not necessarily exact base order.
  std::size_t trials;   ///< Actual starts, fixed forty per original setting.
  std::size_t collided; ///< Actual starts reaching equality before the100000-iteration cap.
  double      mean_iterations; ///< Actual completed Floyd iterations averaged over starts.
  double
      mean_legacy_index; ///< Original zero-based collision loop index average, one less on successful starts.
};
/// @brief Execute the original six-prime/fixed-start first-collision experiment through the shared walk.
/// Uses forty starts per prime, original targets and a0/b0 schedule; reports
/// actual completed iterations separately from the legacy zero-based index.
/// Does not refit a scaling law or count a collision as a successful logarithm.
/// @return Owned six actual summaries, in original modulus order; no shared state.
/// @throws std::length_error If a selected default primitive-call policy is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_rho_log_collision_study().size()); // 6
/// @endcode
std::vector<RhoLogCollisionStudy> pollard_rho_log_collision_study();
} // namespace datamunge::algorithms
