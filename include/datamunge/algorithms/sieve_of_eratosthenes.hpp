#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief The Sieve of Eratosthenes, the classical (c. 240 BC) method for finding every prime up
///        to a bound. Write down the integers 2, 3, ..., n; repeatedly take the smallest number not
///        yet crossed out -- it must be prime -- and cross out all of its multiples. Whatever
///        survives is prime. Marking multiples of each prime p starting at p*p (every smaller
///        multiple carries a smaller prime factor and was already struck) and stopping the outer
///        loop once p*p > n (a composite <= n has a factor <= sqrt(n)) makes the total work
///        O(n log log n) -- the sum of n/p over primes p <= sqrt(n) -- with O(n) memory.
///
///        0 and 1 are not prime by convention; for n < 2 there are no primes.
///
/// @param n inclusive upper bound.
/// @return a bitmask sieve of length n + 1 (empty when n < 2): sieve[i] is true iff i is prime,
///         for every i in 0..n.
std::vector<bool> prime_sieve(std::uint64_t n);

/// @brief Every prime not exceeding @p n, in ascending order -- the true entries of prime_sieve(n)
///        collected into a list. Empty when n < 2. The number of entries is the prime-counting
///        function pi(n), which the Prime Number Theorem places at pi(n) ~ n / ln(n).
///
/// @param n inclusive upper bound.
/// @return the primes p with 2 <= p <= n, ascending.
std::vector<std::uint64_t> primes_up_to(std::uint64_t n);

} // namespace datamunge::algorithms
