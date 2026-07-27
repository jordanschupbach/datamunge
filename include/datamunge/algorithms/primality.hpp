#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief The *Fermat primality test*: a probabilistic test based on Fermat's little theorem, which
///        says that if @p n is prime then @c a^{n-1} ≡ 1 (mod n) for every base @c a coprime to
///        @p n. The test picks @p rounds random bases and declares @p n a *probable prime* if all
///        pass; a single failing base proves compositeness. It is fast but fallible: *Carmichael
///        numbers* (561, 1105, 1729, ...) satisfy the congruence for /every/ coprime base, so the
///        Fermat test calls them prime no matter how many rounds are run -- the flaw that motivated
///        the stronger Miller-Rabin test. (A base sharing a factor with @p n reveals compositeness
///        immediately.)
///
/// @param n      the number to test.
/// @param rounds the number of random bases to try.
/// @param seed   the seed for the internal generator (so results are reproducible).
/// @return false if @p n is definitely composite; true if it is a probable prime for all bases tried.
bool fermat_probable_prime(std::uint64_t n, int rounds = 20, std::uint64_t seed = 0x9E3779B97F4A7C15ULL);

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
///        primes up to @p limit.
///
/// @param limit the inclusive upper bound.
/// @return the primes @c <= limit, in increasing order.
std::vector<std::uint64_t> atkin_primes(std::uint64_t limit);

} // namespace datamunge::algorithms
