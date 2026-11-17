#pragma once
/// @file miller_rabin.hpp
/// @brief Deterministic uint64 primality and actual strong probable-prime rounds.
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned actual sequence of one strong probable-prime round.
/// Early guards have no powering record. For odd n>=3 and a%n!=0, d is odd,
/// n-1=d*2^s, and residues begin with a^d followed by actual post-powering
/// squares until n-1 is reached or all s-1 squares finish. A pass is not a
/// primality certificate. No constant-time guarantee.
/// @par Example
/// @code
/// auto t=dm::algorithms::miller_rabin_round_trace(17,3);
/// std::print("{} {} {}\n",t.d,t.s,t.passed); // 1 4 true
/// @endcode
/// @see miller_rabin_round_trace()
struct MillerRabinRoundTrace {
  std::uint64_t n{0};            ///< Original candidate.
  std::uint64_t base{0};         ///< Original unreduced base.
  std::uint64_t reduced_base{0}; ///< Base modulo n for odd n>=3; otherwise zero.
  std::uint64_t d{0};            ///< Odd part of n-1 when powering occurs; otherwise zero.
  unsigned      s{0};            ///< Factor-of-two exponent when powering occurs; otherwise zero.
  bool          passed{false};   ///< Actual round outcome, including vacuous base-multiple passes.
  ModPowTrace   powering;        ///< Actual shared modular-power trace, empty on early guards.
  std::vector<std::uint64_t> residues; ///< Actual initial residue and successive tested squares.
};
/// @brief Decide primality for the complete unsigned64-bit domain.
/// Screens division by the first twelve primes, then uses these same bases for
/// strong rounds. Their smallest simultaneous strong pseudoprime is
/// 318665857834031151167461, above UINT64_MAX. At most twelve O(log n)
/// fixed-word rounds and O(1) driver storage; no constant-time guarantee.
/// @param n Candidate; zero and one are not prime.
/// @return True exactly when n is prime.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::is_probable_prime(2047),dm::algorithms::is_probable_prime(65537)); // false true
/// @endcode
/// @par References
/// - J. Sorenson and J. Webster (2017), Strong pseudoprimes to twelve prime bases, Mathematics of Computation86,985–1003, doi:10.1090/mcom/3134.
/// @see is_strong_probable_prime_base()
[[nodiscard]] bool is_probable_prime(std::uint64_t n);
/// @brief Perform one actual strong probable-prime round with exact modular arithmetic.
/// For odd n>=3, factor n-1=d*2^s and accept a^d=1 or any tested
/// a^(d*2^r)=n-1 for 0<=r<s. Continue all scheduled squares if a residue
/// becomes one without an earlier minus-one. O(log n) word products, O(1)
/// driver storage. A base divisible by n passes vacuously.
/// @param n Candidate; below2 fails; even candidates pass only for2.
/// @param a Arbitrary base, reduced modulo odd n>=3.
/// @return Actual single-base pass, which can include composites.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::is_strong_probable_prime_base(2047,2),dm::algorithms::is_strong_probable_prime_base(2047,3)); // true false
/// @endcode
/// @par References
/// - M. O. Rabin (1980), Probabilistic algorithm for testing primality, Journal of Number Theory12,128–138, doi:10.1016/0022-314X(80)90084-0.
/// @see miller_rabin_round_trace(), is_probable_prime()
[[nodiscard]] bool is_strong_probable_prime_base(std::uint64_t n, std::uint64_t a);
/// @brief Capture the actual producer's powering and strong-round residues.
/// Same body and early exits as is_strong_probable_prime_base(), without replay.
/// Upfront cap counts powering bit records plus all potentially scheduled strong
/// residues (s); at most64 entries for uint64 candidates. Zero disables the cap.
/// Trace owns its data and uses O(log n) storage; exponent bits are observable.
/// @param n Candidate, with the ordinary early-guard conventions.
/// @param a Arbitrary unreduced base, retained in the trace.
/// @param max_entries Upper-bound record cap, default128; zero disables.
/// @return Owned actual history; unused upper-bound slots are not emitted.
/// @throws std::length_error If the complete upper bound exceeds a nonzero cap, before powering.
/// @throws std::bad_alloc If capture allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::miller_rabin_round_trace(561,2);
/// std::print("{} {} {}\n",t.d,t.s,t.passed); // 35 4 false
/// @endcode
/// @par References
/// - M. O. Rabin (1980), doi:10.1016/0022-314X(80)90084-0.
/// @see MillerRabinRoundTrace, is_strong_probable_prime_base()
[[nodiscard]] MillerRabinRoundTrace
miller_rabin_round_trace(std::uint64_t n, std::uint64_t a, std::size_t max_entries = 128);
} // namespace datamunge::algorithms
