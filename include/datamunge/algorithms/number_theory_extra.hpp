#pragma once

#include <cstdint>
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

/// @brief *Fermat's factorization method*: writes an odd @p n as a *difference of squares*
///        @c n = a^2 - b^2 = (a-b)(a+b). Starting at @c a = ceil(sqrt(n)) it increments @c a until
///        @c a^2 - n is a perfect square @c b^2, then reads off the factors. It is very fast when
///        @p n has two factors *close together* (few steps), and slow when they are far apart -- the
///        exact complementary regime to trial division and Pollard's rho.
///
/// @param n an odd integer to factor.
/// @return a nontrivial factor of @p n, or @p n itself if none is found (e.g. @p n prime).
std::uint64_t fermat_factor(std::uint64_t n);

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

/// @brief *Dixon's factorization algorithm*: the archetype of modern *congruence-of-squares*
///        factoring (the family that includes the quadratic sieve and the number field sieve). It
///        looks for integers @c x whose square mod @p n is *smooth* -- factors completely over a small
///        *factor base* of primes -- collects enough such relations that a subset multiplies to a
///        perfect square on both sides, and thereby builds a congruence @c X^2 ≡ Y^2 (mod n) with
///        @c X != +-Y, so @c gcd(X-Y, n) is a nontrivial factor. The subset is found by Gaussian
///        elimination over GF(2) on the exponent-parity vectors.
///
/// @param n the odd composite to factor.
/// @param base_size the number of small primes to use in the factor base.
/// @return a nontrivial factor of @p n, or @c 0 if none was found with this factor base.
std::uint64_t dixon_factor(std::uint64_t n, int base_size);

} // namespace datamunge::algorithms
