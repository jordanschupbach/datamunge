#pragma once
/// @file number_theory_extra.hpp
/// @brief Trial division, exact bounded Fermat splitting and Pollard p-minus-one factorization.

#include <datamunge/algorithms/dixon.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Trial division*: the elementary integer-factorization method -- try every candidate
///        divisor @c 2,3,5,7,... up to @c sqrt(n), dividing it out (with multiplicity) whenever it
///        divides. Any factor left above @c sqrt(n) is prime. It is @c O(sqrt(n)) and hopeless for
///        large semiprimes, but it is exact, dead simple, and the right tool for small numbers or for
///        peeling off the small prime factors before a heavier method takes over.
///
/// @param n the integer to factor (@c n >= 1).
/// @return the prime factors of @p n in nondecreasing order, with multiplicity (empty for @c n=1).
std::vector<std::uint64_t> trial_division_factorize(std::uint64_t n);

/// @brief Reason that a bounded Fermat factorization stopped.
/// Values describe actual work, not a probabilistic primality test. TrivialPair
/// reaches the first odd difference-of-squares pair with smaller factor one;
/// TrialLimit makes no primality claim. Even bypasses square testing.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fermat_stop_name(dm::algorithms::FermatStop::TrialLimit)); // trial limit
/// @endcode
/// @see fermat_factor_work(), FermatOutcome
enum class FermatStop {
    Even,        ///< Input is even; returned factor two, with no square tests.
    Factor,      ///< An actual square test found a proper split, possibly composite factors.
    TrivialPair, ///< Odd search reached factors one and n after excluding closer pairs.
    TrialLimit   ///< Positive trial budget ended before a square hit; no primality conclusion.
};

/// @brief Owned result and actual work counts of bounded Fermat splitting.
/// For odd n, start_a is the exact ceiling root; last_a and last_b are the last
/// actual trial and its exact floor root of a squared minus n. Trials count square
/// tests, including a hit; increments count moves between tested a values, hence
/// trials-1. An even input has zero trials, increments and a/b metadata.
/// Returned_factor matches fermat_factor(); cofactor is n/returned_factor when a
/// square or even branch completes, and zero at TrialLimit. Proper_split requires
/// both factors greater than one. Results own their scalars and share no state.
/// @par Example
/// @code
/// const auto f=dm::algorithms::fermat_factor_work(5959);
/// std::print("{} {} {}\n",f.returned_factor,f.cofactor,f.trials); // 59 101 3
/// @endcode
/// @see FermatStop, FermatTrace, fermat_factor_work()
struct FermatOutcome {
    std::uint64_t n=0;               ///< Original input, at least two in a returned outcome.
    std::uint64_t returned_factor=0; ///< Proper smaller factor, two for even input, or n if unsplit.
    std::uint64_t cofactor=0;        ///< Known complementary factor; zero only at TrialLimit.
    std::uint64_t start_a=0;         ///< Exact ceiling square root for odd inputs; zero for even.
    std::uint64_t last_a=0;          ///< Last actually tested a; zero for even inputs.
    std::uint64_t last_b=0;          ///< Exact floor root of last difference; zero for even inputs.
    bool proper_split=false;        ///< Both returned factors are strictly greater than one.
    FermatStop stop=FermatStop::TrialLimit; ///< Actual termination branch.
    std::size_t trials=0;           ///< Actual difference-of-squares tests, including a successful test.
    std::size_t increments=0;       ///< Moves between actual trials, excluding an unused next a.
};

/// @brief One actual odd Fermat square test, owned without input references.
/// Difference is computed in unsigned 128-bit arithmetic before its exact integer
/// root b. Square reports equality of b squared and difference; b is not rounded
/// from floating arithmetic. Rows come from the solver, never a replay loop.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fermat_factor_trace(5959);
/// std::print("{} {}\n",t.rows.back().a,t.rows.back().b); // 80 21
/// @endcode
/// @see FermatTrace, integer_sqrt_u128()
struct FermatStep {
    std::uint64_t a=0;              ///< Actual integer being tested.
    unsigned __int128 difference=0; ///< Exact a squared minus n, possibly wider than uint64.
    std::uint64_t b=0;              ///< Exact floor square root of difference.
    bool square=false;             ///< Whether b squared equals difference.
};

/// @brief Owned same-pass Fermat outcome and optional actual trial history.
/// C=captured_entries=rows.size() counts whole FermatStep records, not scalar
/// fields or bytes. Quota admission occurs before square/root work and storing
/// each row. Input/output scalars, vector capacity, allocator and runtime storage
/// are excluded: C is not peak RAM. Copies are independent; editing a row does not
/// recompute outcome. Even inputs capture nothing. Ordinary/work calls store no
/// optional history and impose no history quota.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fermat_factor_trace(5959,3,3);
/// std::print("{} {}\n",t.captured_entries,t.outcome.increments); // 3 2
/// @endcode
/// @see fermat_factor_trace(), FermatStep
struct FermatTrace {
    FermatOutcome outcome;               ///< Actual bounded solver outcome.
    std::vector<FermatStep> rows;         ///< Owned actual trial records in execution order.
    std::size_t captured_entries=0;       ///< Number of admitted/stored whole records.
};

/// @brief Split n by exact Fermat square tests with the historical five-million-trial cap.
/// The original compiled signature is retained and delegates to the same kernel
/// as fermat_factor_work(). Even inputs return two immediately. Odd inputs test
/// successive a from the exact ceiling root; a square gives a-b when it is a
/// proper factor, otherwise n. Returning n also occurs on budget exhaustion and
/// does not prove primality. The first factors need not be prime. Uses exact
/// uint128 differences/roots, O(T) fixed-word work and constant auxiliary storage;
/// at most 63 division comparisons per 128-bit root, not constant arbitrary-bit
/// complexity. Independent calls share no mutable state.
/// @param n Integer at least two; zero and one are rejected.
/// @return Two for even n; a proper smaller factor if found, otherwise n.
/// @throws std::invalid_argument If n is less than two.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fermat_factor(5959)); // 59
/// @endcode
/// @par References
/// - C. Pomerance (1996), A Tale of Two Sieves, Notices AMS 43(12), pp.1473-1485,
///   Fermat and Kraitchik section: https://math.dartmouth.edu/~carlp/PDF/paper109.pdf
///   Uses the difference-of-squares identity with an explicit finite trial cap.
/// @see fermat_factor_work(), fermat_factor_trace(), integer_sqrt_u128()
std::uint64_t fermat_factor(std::uint64_t n);

/// @brief Run bounded exact Fermat splitting and count its actual tests and increments.
/// Shares the ordinary solver's kernel and constant storage with no optional
/// history or history quota. A hit is one test, so odd successful calls take
/// increments+1 tests. Even n bypasses square tests, including n=2 which has no
/// proper split. TrivialPair excludes all earlier odd factor pairs and proves
/// odd n prime; TrialLimit is inconclusive. Positive controls are validated before
/// the even branch. Fixed-word work is O(trials), each root at most63 comparisons.
/// @param n Integer at least two, including the full uint64 range.
/// @param max_trials Positive maximum square tests; defaults to5000000.
/// @return Owned factors, stop reason, last actual state and work counts.
/// @throws std::invalid_argument If n<2 or max_trials is zero.
/// @par Example
/// @code
/// const auto f=dm::algorithms::fermat_factor_work(5959,2);
/// std::print("{} {} {}\n",f.returned_factor,f.trials,dm::algorithms::fermat_stop_name(f.stop)); // 5959 2 trial limit
/// @endcode
/// @par References
/// - C. Pomerance (1996), A Tale of Two Sieves, Fermat and Kraitchik section:
///   https://math.dartmouth.edu/~carlp/PDF/paper109.pdf
/// @see fermat_factor(), FermatOutcome, fermat_factor_trace()
FermatOutcome fermat_factor_work(std::uint64_t n,std::size_t max_trials=5000000);

/// @brief Capture actual Fermat trials during the same bounded factorization pass.
/// No replay: outcome is identical to work(n,max_trials) if capture succeeds.
/// Before each square/root test, admit one whole record against max_entries.
/// Default quota1000000 and default trial budget5000000 are independent. Zero
/// max_entries disables only the optional quota; max_trials must stay positive.
/// A quota/allocation failure throws without returning a partial trace. No shared
/// mutable state; owned copies survive other calls and edits independently.
/// Costs O(T) fixed-word work and O(T) records, unlike ordinary constant storage.
/// @param n Integer at least two.
/// @param max_trials Positive maximum actual square tests; defaults to5000000.
/// @param max_entries Maximum captured whole rows; defaults to1000000; zero disables quota.
/// @return Owned outcome and one row per actual odd square test; none for even n.
/// @throws std::invalid_argument If n<2 or max_trials is zero, before capture.
/// @throws std::length_error Before a row would exceed quota or vector maximum size.
/// @throws std::bad_alloc If history storage allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fermat_factor_trace(5959,3,0);
/// std::print("{} {}\n",t.rows.size(),t.outcome.returned_factor); // 3 59
/// @endcode
/// @par References
/// - C. Pomerance (1996), A Tale of Two Sieves, Fermat and Kraitchik section:
///   https://math.dartmouth.edu/~carlp/PDF/paper109.pdf
/// @see fermat_factor_work(), FermatTrace
FermatTrace fermat_factor_trace(std::uint64_t n,std::size_t max_trials=5000000,std::size_t max_entries=1000000);

/// @brief Return stable descriptive text for a Fermat termination reason.
/// Literal string views have static lifetime and no allocation or shared mutation.
/// Invalid enum representations give "unknown".
/// @param stop Stop reason to describe.
/// @return "even", "factor", "trivial pair", "trial limit", or "unknown".
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fermat_stop_name(dm::algorithms::fermat_factor_work(13).stop)); // trivial pair
/// @endcode
/// @see FermatStop, fermat_factor_work()
std::string_view fermat_stop_name(FermatStop stop) noexcept;

/// @brief *Pollard's p-1 algorithm*: finds a prime factor @c p of @p n for which @c p-1 is
///        *B-smooth* (all its prime factors are @c <= B). It computes @c a^M mod n for a highly
///        composite exponent @c M (the product of prime powers up to @p bound); by Fermat's little
///        theorem @c a^M ≡ 1 (mod p) once @c (p-1) | M, so @c gcd(a^M - 1, n) exposes @c p. It is the
///        method that motivates using *strong* (safe) primes in cryptography, where @c p-1 has a large
///        prime factor and this attack fails.
///
/// @param n the integer to factor.
/// @param bound the smoothness bound @c B.
/// @return a nontrivial factor of @p n, or @c 0 if the method fails for this @p bound.
std::uint64_t pollard_p_minus_1(std::uint64_t n, std::uint64_t bound);


} // namespace datamunge::algorithms
