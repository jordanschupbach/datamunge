#pragma once
/// @file pollard_kangaroo.hpp
/// @brief Native bounded-interval kangaroo jumps with exact widened distance histories.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Historical owned verified candidate and actual total jump count.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_kangaroo(2,8,1019,3,3).x); // 3
/// @endcode
struct KangarooResult {
  std::uint64_t x{0}; ///< Verified exponent in [a,b] when found; zero placeholder otherwise.
  bool          found{false}; ///< True only after an actual modular-power equality check.
  std::uint64_t jumps{
      0}; ///< Actual tame plus wild jumps across all attempts; endpoint shortcuts use zero.
};
/// @brief Actual-jump and retained-record policies; zero disables each cap.
/// @par Example
/// @code
/// const dm::algorithms::KangarooOptions o;std::print("{}\n",o.max_attempts); // 8
/// @endcode
struct KangarooOptions {
  std::size_t max_jumps{
      1000000}; ///< Total actual jump cap, also admits each planned complete tame phase before reserving its trail.
  std::size_t max_trace_records{
      1000000}; ///< Jump-table/jump/matched-collision records; not bytes or peak working trail memory.
  std::size_t max_attempts{
      8}; ///< Native seeded attempts; zero still checks interval endpoints before returning unresolved.
};
/// @brief Selected actual call-boundary and working-trail diagnostics.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_kangaroo_run(2,8,1019,3,3).attempts); // 0
/// @endcode
struct KangarooRun {
  KangarooResult result;      ///< Historical verified outcome and actual total jumps.
  std::size_t    attempts{0}; ///< Actual jump tables/complete tame trails started.
  std::size_t power_calls{0}; ///< Actual canonical mod_pow calls; internal bit products excluded.
  std::size_t tame_assignments{
      0}; ///< Actual trail assignments including initial b; includes overwrites.
  std::size_t trail_overwrites{
      0};                      ///< Actual assignments replacing a previously visited group point.
  std::size_t wild_lookups{0}; ///< Actual hash-map lookup calls before wild jumps.
};
/// @brief One actual seeded jump-table entry, as used by both kangaroos.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_kangaroo_trace(2,8,1019,0,10);std::print("{}\n",t.table[0].step>0); // true
/// @endcode
struct KangarooJumpSize {
  std::size_t   attempt; ///< Zero-based native attempt.
  unsigned      index;   ///< Table slot in [0,31].
  std::uint64_t step;    ///< Actual positive exponent jump.
  std::uint64_t power;   ///< Actual g^step mod P precomputation.
};
/// @brief One actual tame or wild group jump and exact distance.
/// The native-only unsigned128 distance avoids b+distance wrap at uint64 limits.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_kangaroo_trace(2,8,1019,0,10);std::print("{}\n",t.jumps[0].wild); // false
/// @endcode
struct KangarooJump {
  std::size_t   attempt; ///< Actual attempt id.
  bool          wild;    ///< False for tame trail building, true for target-starting walk.
  std::uint64_t before;  ///< Actual pre-jump group element.
  std::uint64_t after;   ///< Actual post-jump group element.
  unsigned      slot;    ///< Actual before modulo32 table selector.
  std::uint64_t step;    ///< Actual exponent jump used.
  unsigned __int128
      distance_after; ///< Exact cumulative tame/wild distance since that walk's start.
  unsigned __int128
      trail_exponent_after; ///< Actual assigned b+distance for a tame jump; zero sentinel for a wild jump.
};
/// @brief One actual successful trail lookup and candidate check.
/// @par Example
/// @code
/// const dm::algorithms::KangarooCollision c{};std::print("{}\n",c.verified); // false
/// @endcode
struct KangarooCollision {
  std::size_t       attempt;       ///< Actual attempt id.
  std::uint64_t     point;         ///< Actual matching group point before a wild jump.
  unsigned __int128 tame_exponent; ///< Exact most recently assigned b+tame-distance for this point.
  unsigned __int128 wild_distance; ///< Exact current wild distance.
  bool              in_interval;   ///< Difference is nonnegative and lies within [a,b].
  std::uint64_t     candidate; ///< Actual difference when in_interval; zero placeholder otherwise.
  bool verified; ///< True when the actual candidate modular power matches normalized h.
};
/// @brief Owned actual native tables, jumps and matched-lookup outcomes.
/// @par Example
/// @code
/// const auto t=dm::algorithms::pollard_kangaroo_trace(2,8,1019,0,10);std::print("{}\n",t.jumps.size()==t.run.result.jumps); // true
/// @endcode
struct KangarooTrace {
  KangarooRun                    run;   ///< Ordinary producing result and actual selected work.
  std::vector<KangarooJumpSize>  table; ///< Actual per-attempt32-entry seeded tables.
  std::vector<KangarooJump>      jumps; ///< Actual producing tame then wild jumps.
  std::vector<KangarooCollision> collisions; ///< Actual matched trail lookups and candidate checks.
};
/// @brief Attempt g^x=h modulo P for a<=x<=b with native deterministic kangaroos.
/// Reduces g,h modulo P and requires both to be units; composite P is allowed,
/// but no cyclicity, exact base-order or target-membership proof is supplied.
/// Each attempt builds the complete original32-jump seeded tame trail before
/// following the wild walk. Trail collisions can reflect group-period aliases;
/// every candidate is range/equation checked. Failure after finite attempts is
/// not a proof no exponent exists. Exact unsigned128 distances avoid wrapping
/// near UINT64_MAX; returned x still fits the requested uint64 interval.
/// Input is scalar, state is local. Expected square-root behavior is heuristic;
/// actual selected work and trail storage are capped by default.
/// @param g Base, reduced modulo P; gcd(g,P) must be one.
/// @param h Target, reduced modulo P; gcd(h,P) must be one.
/// @param P Modulus >=2, prime not required.
/// @param a Inclusive lower exponent bound.
/// @param b Inclusive upper bound; b<a returns unresolved after complete modulus/unit validation.
/// @param seed Original native table seed, including zero.
/// @return Owned verified exponent/found flag and actual tame+wild jump count.
/// @throws std::invalid_argument If modulus or unit inputs are invalid.
/// @throws std::length_error If default jump policy or representable trail capacity is exceeded.
/// @throws std::overflow_error If actual call/work counters cannot be represented.
/// @par Example
/// @code
/// const auto r=dm::algorithms::pollard_kangaroo(2,8,1019,3,3);std::print("{} {}\n",r.found,r.x); // true 3
/// @endcode
/// @par References
/// - J. M. Pollard (1978), Monte Carlo methods for index computation (mod p),
///   doi:10.1090/S0025-5718-1978-0491431-9. The32-entry, eight-attempt trail is the native variant.
KangarooResult pollard_kangaroo(
    std::uint64_t g,
    std::uint64_t h,
    std::uint64_t P,
    std::uint64_t a,
    std::uint64_t b,
    std::uint64_t seed = 0x2545F4914F6CDD1DULL);
/// @brief Run the shared actual native kangaroos with selected diagnostics and policies.
/// @param g Unit base reduced modulo P.
/// @param h Unit target reduced modulo P.
/// @param P Modulus >=2.
/// @param a Inclusive lower bound.
/// @param b Inclusive upper bound.
/// @param seed Native deterministic jump-table seed.
/// @param options Jump/history policies and attempt count; history policy unused here.
/// @return Owned ordinary result and selected actual work, without historical records.
/// @throws std::invalid_argument If modulus or units are invalid.
/// @throws std::length_error If jump policy or trail capacity is exceeded.
/// @throws std::overflow_error If actual counters cannot be represented.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_kangaroo_run(2,8,1019,3,3).result.found); // true
/// @endcode
KangarooRun pollard_kangaroo_run(
    std::uint64_t   g,
    std::uint64_t   h,
    std::uint64_t   P,
    std::uint64_t   a,
    std::uint64_t   b,
    std::uint64_t   seed    = 0x2545F4914F6CDD1DULL,
    KangarooOptions options = {});
/// @brief Capture actual tables, jumps and collision checks without replaying either walk.
/// O(actual jumps+matched lookups+32*attempts) retained records beyond working
/// trails. History contains exact native-only unsigned128 distances.
/// @param g Unit base modulo P.
/// @param h Unit target modulo P.
/// @param P Modulus >=2.
/// @param a Inclusive lower bound.
/// @param b Inclusive upper bound.
/// @param seed Native deterministic jump-table seed.
/// @param options Jump/history policies and finite attempt count.
/// @return Owned actual outcome, native tables and complete producing records.
/// @throws std::invalid_argument If modulus or units are invalid.
/// @throws std::length_error If jump/history policies or trail capacity are exceeded.
/// @throws std::overflow_error If actual counters cannot be represented.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pollard_kangaroo_trace(2,8,1019,3,3).jumps.size()); // 0
/// @endcode
KangarooTrace pollard_kangaroo_trace(
    std::uint64_t   g,
    std::uint64_t   h,
    std::uint64_t   P,
    std::uint64_t   a,
    std::uint64_t   b,
    std::uint64_t   seed    = 0x2545F4914F6CDD1DULL,
    KangarooOptions options = {});
} // namespace datamunge::algorithms
