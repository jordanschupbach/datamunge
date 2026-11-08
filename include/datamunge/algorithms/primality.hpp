#pragma once

/// @file primality.hpp
/// @brief Primality tests (Fermat, Miller-Rabin, Solovay-Strassen, AKS and others) and prime sieves.

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {

/// @brief Actual stop reason for the seeded Fermat congruence test.
/// BelowTwo means not prime (zero and one are not composite). SmallPrime covers
/// two and three; Even rejects larger even inputs. Witness is a failed actual
/// congruence; RoundLimit means all requested rounds passed, not a proof.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fermat_primality_stop_name(dm::algorithms::FermatPrimalityStop::RoundLimit)); // rounds passed
/// @endcode
/// @see FermatPrimalityOutcome, fermat_probable_prime_work()
enum class FermatPrimalityStop {
    BelowTwo,   ///< Zero or one: not prime, with no drawn bases.
    SmallPrime, ///< Two or three: known prime, with no drawn bases.
    Even,       ///< Even input greater than two: composite, with no drawn bases.
    Witness,    ///< Actual tested base failed the congruence, proving compositeness.
    RoundLimit  ///< All requested positive rounds passed; no primality certificate.
};

/// @brief Owned exact result of one Fermat congruence and its actual gcd.
/// Tests base^(n-1) modulo n using canonical mod_pow, plus std::gcd. A failure
/// proves n composite; a pass does not establish primality. Noncoprime bases
/// cannot pass. No shared state, retained references or floating arithmetic.
/// @par Example
/// @code
/// const auto r=dm::algorithms::fermat_base_test(15,2);
/// std::print("{} {} {}\n",r.common_divisor,r.residue,r.passes); // 1 4 false
/// @endcode
/// @see fermat_base_test(), FermatPrimalityTrace
struct FermatBaseResult {
    std::uint64_t n=0;              ///< Original modulus, at least two in a returned record.
    std::uint64_t base=0;           ///< Original tested base, in [1,n).
    std::uint64_t common_divisor=0; ///< Actual gcd(base,n); a proper factor when greater than one.
    std::uint64_t residue=0;        ///< Exact base^(n-1) modulo n.
    bool passes=false;             ///< Whether the actual residue equals one.
};

/// @brief Owned result, actual round count and last state of seeded Fermat testing.
/// Requested rounds stay separate from completed rounds; early filters perform
/// zero congruences. Last base/gcd/residue are zero for filters, otherwise actual
/// values from the last tested round. True means prime for SmallPrime and only
/// probable prime for RoundLimit. Scalar ownership has no shared mutable state.
/// @par Example
/// @code
/// const auto o=dm::algorithms::fermat_probable_prime_work(97,3);
/// std::print("{} {}\n",o.probable_prime,o.completed_rounds); // true 3
/// @endcode
/// @see FermatPrimalityStop, fermat_probable_prime_work()
struct FermatPrimalityOutcome {
    std::uint64_t n=0;              ///< Original unsigned input, including zero/one.
    std::uint64_t seed=0;           ///< Original SplitMix64 seed.
    int requested_rounds=0;         ///< Positive requested maximum drawn bases.
    int completed_rounds=0;         ///< Actual congruences tested, including a failing one.
    bool probable_prime=false;     ///< Result shared with fermat_probable_prime().
    FermatPrimalityStop stop=FermatPrimalityStop::BelowTwo; ///< Actual termination branch.
    std::uint64_t last_base=0;      ///< Last actual base, or zero for a filter.
    std::uint64_t last_gcd=0;       ///< Last actual gcd, or zero for a filter.
    std::uint64_t last_residue=0;   ///< Last actual power residue, or zero for a filter.
};

/// @brief Owned actual seeded round records captured during one Fermat test.
/// C=captured_entries=rows.size() counts whole FermatBaseResult records. The
/// solver admits a record before drawing its base or computing gcd/power; no
/// replay and no nested exponent-bit history. Input/output scalars, capacity,
/// allocators and runtime storage are excluded, so C is not peak RAM. Copies
/// are independent; editing rows does not recompute the stored outcome.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fermat_probable_prime_trace(97,3);
/// std::print("{} {}\n",t.rows.size(),t.captured_entries); // 3 3
/// @endcode
/// @see fermat_probable_prime_trace(), FermatBaseResult
struct FermatPrimalityTrace {
    FermatPrimalityOutcome outcome;  ///< Actual result and completed-round count.
    std::vector<FermatBaseResult> rows; ///< Actual drawn bases and results in order, stopping at a witness.
    std::size_t captured_entries=0;  ///< Number of whole round records admitted and stored.
};

/// @brief Exact exhaustive counts over a specified half-open Fermat base interval.
/// Counts are integer census results, not samples or posterior probabilities.
/// Passes are necessarily coprime, so passes/coprime is the conditional pass
/// fraction only when coprime>0. Passes/total is the unconditioned interval
/// fraction only when total>0. Empty intervals have all counts zero; no fraction
/// is fabricated. Owned scalars and constant driver storage, with no shared state.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fermat_base_census(15,1,15);
/// std::print("{} {} {}\n",c.total,c.coprime,c.passes); // 14 8 4
/// @endcode
/// @see fermat_base_census(), fermat_base_test()
struct FermatBaseCensus {
    std::uint64_t n=0;        ///< Original modulus, at least two.
    std::uint64_t first=0;    ///< Inclusive interval lower base.
    std::uint64_t last=0;     ///< Exclusive interval upper base, at most n.
    std::uint64_t total=0;    ///< Actual bases tested, exactly last-first.
    std::uint64_t coprime=0;  ///< Actual bases with gcd(base,n)=1.
    std::uint64_t passes=0;   ///< Actual residues equal to one; all are among coprime bases.
};

/// @brief Run the seeded Fermat congruence test, preserving its original compiled signature.
/// Uses SplitMix64 and a=2+word%(n-3) for odd n>=5, stopping at the first failed
/// residue. This samples all bases in [2,n-2], including noncoprime bases; hence
/// Carmichael numbers can be rejected. Fixed seed means deterministic output;
/// modulo reduction is generally biased and draws are not guaranteed independent
/// uniform samples. No 2^-rounds error guarantee or cryptographic RNG claim.
/// Uses canonical exact mod_pow and one gcd per actual round, O(R log(n))
/// fixed-word arithmetic and constant auxiliary storage; no optional history.
/// @param n Unsigned integer to test; zero/one give false, two/three true, larger evens false.
/// @param rounds Positive maximum bases; defaults to20; validated before input filters.
/// @param seed Initial SplitMix64 state; default0x9E3779B97F4A7C15ULL.
/// @return False on a not-prime filter or witness; true on small primes or all rounds passing.
/// @throws std::invalid_argument If rounds is nonpositive, before any filtering/draw.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::fermat_probable_prime(97),dm::algorithms::fermat_probable_prime(561)); // true false
/// @endcode
/// @par References
/// - K. Conrad, Fermat's Test, sections1-3, https://kconrad.math.uconn.edu/blurbs/ugradnumthy/fermattest.pdf
///   Uses a reproducible finite base stream instead of ideal independent uniform draws.
/// @see fermat_probable_prime_work(), fermat_base_test(), fermat_base_census()
bool fermat_probable_prime(std::uint64_t n, int rounds = 20, std::uint64_t seed = 0x9E3779B97F4A7C15ULL);

/// @brief Test one specified base using an exact power and gcd, with no sampling.
/// A failing residue proves compositeness; a common_divisor greater than one is
/// additionally a proper factor because 1<=base<n. Passing bases, including one,
/// do not prove primality. Canonical mod_pow uses full128bit products; O(log(n))
/// fixed-word work and constant storage. Does not mutate shared state.
/// @param n Modulus at least two, spanning the full uint64 range.
/// @param base Integer in [1,n); not silently reduced from an out-of-range input.
/// @return Owned input/gcd/residue and congruence result.
/// @throws std::invalid_argument If n<2, base==0, or base>=n, before modular arithmetic.
/// @par Example
/// @code
/// const auto r=dm::algorithms::fermat_base_test(561,3);
/// std::print("{} {}\n",r.common_divisor,r.passes); // 3 false
/// @endcode
/// @par References
/// - K. Conrad, Fermat's Test, https://kconrad.math.uconn.edu/blurbs/ugradnumthy/fermattest.pdf
/// @see mod_pow(), FermatBaseResult, fermat_base_census()
FermatBaseResult fermat_base_test(std::uint64_t n,std::uint64_t base);

/// @brief Run the original seeded base stream and expose actual completed rounds and stop.
/// Shares one kernel with the bool and trace calls; no optional allocation or
/// capture quota. Filters draw no bases; a witness round counts as completed.
/// Last-state scalars are actual values, never reconstructed. Seeded modulo
/// sampling is deterministic and does not supply an iid error bound.
/// @param n Unsigned input; zero/one are not prime; two/three known prime; larger evens composite.
/// @param rounds Positive maximum rounds, default20, validated before filters.
/// @param seed Initial SplitMix64 state, default0x9E3779B97F4A7C15ULL.
/// @return Owned actual decision, stop, count and last round state.
/// @throws std::invalid_argument If rounds<=0.
/// @par Example
/// @code
/// const auto o=dm::algorithms::fermat_probable_prime_work(4);
/// std::print("{} {} {}\n",o.probable_prime,o.completed_rounds,dm::algorithms::fermat_primality_stop_name(o.stop)); // false 0 even
/// @endcode
/// @par References
/// - K. Conrad, Fermat's Test, https://kconrad.math.uconn.edu/blurbs/ugradnumthy/fermattest.pdf
/// @see fermat_probable_prime(), FermatPrimalityOutcome
FermatPrimalityOutcome fermat_probable_prime_work(std::uint64_t n,int rounds=20,std::uint64_t seed=0x9E3779B97F4A7C15ULL);

/// @brief Capture actual drawn bases and their exact results during the same seeded test.
/// Outcome and prefix match ordinary/work calls when capture succeeds. Admits
/// one whole round record before the draw, gcd and power. Quota defaults to1m;
/// zero disables only capture quota. Nonpositive rounds remain invalid. Capture
/// failures throw without returning a partial trace. No mutable global RNG;
/// results and copies own all storage. O(R log(n)) word work and O(R) records.
/// @param n Arbitrary unsigned input, with the ordinary zero-work filters.
/// @param rounds Positive maximum rounds, default20.
/// @param seed Initial SplitMix64 state, default0x9E3779B97F4A7C15ULL.
/// @param max_entries Maximum actual whole round records, default1000000; zero disables quota.
/// @return Owned actual outcome and executed round prefix, including a witness if found.
/// @throws std::invalid_argument If rounds<=0, before filters/quota checks.
/// @throws std::length_error Before the next actual round exceeds quota or vector maximum size.
/// @throws std::bad_alloc If history storage allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fermat_probable_prime_trace(97,2,0,2);
/// std::print("{} {}\n",t.outcome.probable_prime,t.captured_entries); // true 2
/// @endcode
/// @par References
/// - K. Conrad, Fermat's Test, https://kconrad.math.uconn.edu/blurbs/ugradnumthy/fermattest.pdf
/// @see fermat_probable_prime_work(), FermatPrimalityTrace
FermatPrimalityTrace fermat_probable_prime_trace(std::uint64_t n,int rounds=20,std::uint64_t seed=0x9E3779B97F4A7C15ULL,std::size_t max_entries=1000000);

/// @brief Count exact Fermat passes and coprime bases over a bounded explicit interval.
/// Evaluates every base once via fermat_base_test, including noncoprime bases.
/// Use [2,n) for the original report's denominator (coprime bases excluding one),
/// [2,n-1) for the sampler support of odd n>=5, or [1,n) for the full unit-group
/// discussion. Default max_bases1m limits total bases before any arithmetic;
/// zero disables that work cap. Empty [n,n) is valid. No allocation or history;
/// O((last-first) log(n)) word work, constant auxiliary storage. Full ranges for
/// large n are intentionally expensive; a tiny high-base interval is supported.
/// @param n Modulus at least two.
/// @param first Inclusive first base, with1<=first<=last.
/// @param last Exclusive upper base, with last<=n.
/// @param max_bases Maximum bases to test, default1000000; zero disables work cap.
/// @return Owned exact interval/count scalars; empty intervals have zero counts.
/// @throws std::invalid_argument If n<2 or interval violates1<=first<=last<=n.
/// @throws std::length_error If interval size exceeds a nonzero max_bases, before powers/gcds.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fermat_base_census(91,2,91);
/// std::print("{} {} {}\n",c.total,c.coprime,c.passes); // 89 71 35
/// @endcode
/// @par References
/// - K. Conrad, Fermat's Test, https://kconrad.math.uconn.edu/blurbs/ugradnumthy/fermattest.pdf
/// @see fermat_base_test(), FermatBaseCensus
FermatBaseCensus fermat_base_census(std::uint64_t n,std::uint64_t first,std::uint64_t last,std::uint64_t max_bases=1000000);

/// @brief Describe a seeded Fermat stop using a static literal string view.
/// No allocation or shared mutation. Invalid enum representations give "unknown".
/// @param stop Actual reason to describe.
/// @return "below two", "small prime", "even", "witness", "rounds passed", or "unknown".
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fermat_primality_stop_name(dm::algorithms::FermatPrimalityStop::Witness)); // witness
/// @endcode
/// @see FermatPrimalityStop, fermat_probable_prime_work()
std::string_view fermat_primality_stop_name(FermatPrimalityStop stop) noexcept;

/// @brief The *Lucas primality test* (Lucas-Lehmer-style converse of Fermat, via Lehmer): a
///        *deterministic* primality proof. @p n (> 1) is prime iff there exists a base @c a with
///        @c a^{n-1} ≡ 1 (mod n) and @c a^{(n-1)/q} ≢ 1 (mod n) for every prime factor @c q of
///        @c n-1 -- i.e. @c a has multiplicative order exactly @c n-1, so the group of units is
///        cyclic of order @c n-1, which forces @p n prime. This implementation factors @c n-1 by
///        trial division and searches for such a witness @c a; finding one proves primality,
///        finding none (after all bases) proves compositeness. Unlike Fermat's test it never errs,
///        at the cost of needing the factorization of @c n-1.
///
/// @param n the number to test.
/// @return true iff @p n is prime.
bool lucas_primality_test(std::uint64_t n);

/// @brief The *Sieve of Sundaram* (Sundaram, 1934): generates all primes up to @p limit. It sieves
///        the integers @c i+j+2ij (for @c 1 <= i <= j); the numbers @c k that are *not* of that form
///        map to the odd primes @c 2k+1. It uses about half the space of the Sieve of Eratosthenes
///        (it never represents even numbers) and runs in @c O(limit log limit).
///
/// @param limit the inclusive upper bound.
/// @return the primes @c <= limit, in increasing order.
std::vector<std::uint64_t> sundaram_primes(std::uint64_t limit);

/// @brief The *Sieve of Atkin* (Atkin & Bernstein, 2003): a modern prime sieve that marks candidates
///        by counting solutions to three quadratic forms modulo 60 (@c 4x^2+y^2, @c 3x^2+y^2, and
///        @c 3x^2-y^2 with the appropriate residues), toggling each hit, then removes squares of
///        primes. It has a better asymptotic operation count than Eratosthenes -- @c O(limit) with
///        wheel optimization (here @c O(limit) toggles over the quadratics) -- and generates all
///        primes up to @p limit. Each form is enumerated only over its own region, about
///        @c 1.04*limit lattice points in all. Defined in sieve_of_atkin.cpp; see
///        sieve_of_atkin.hpp for the forms and operation counts.
///
/// @param limit the inclusive upper bound.
/// @return the primes @c <= limit, in increasing order.
std::vector<std::uint64_t> atkin_primes(std::uint64_t limit);

} // namespace datamunge::algorithms
