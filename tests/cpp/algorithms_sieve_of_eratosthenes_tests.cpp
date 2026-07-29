#include <gtest/gtest.h>

#include <datamunge/algorithms/sieve_of_eratosthenes.hpp>

#include <cstdint>
#include <vector>

using datamunge::algorithms::prime_sieve;
using datamunge::algorithms::primes_up_to;

namespace {

// Independent O(sqrt(n)) trial-division primality test, used only to cross-check the sieve.
bool is_prime_trial(std::uint64_t x) {
    if (x < 2) return false;
    if (x < 4) return true;            // 2 and 3
    if (x % 2 == 0) return false;
    for (std::uint64_t d = 3; d * d <= x; d += 2)
        if (x % d == 0) return false;
    return true;
}

std::size_t pi(std::uint64_t n) { return primes_up_to(n).size(); }

} // namespace

TEST(SieveOfEratosthenes, PrimesUpTo30) {
    const std::vector<std::uint64_t> expected = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    EXPECT_EQ(primes_up_to(30), expected);
}

TEST(SieveOfEratosthenes, PrimeCountingFunctionAtKnownPoints) {
    // pi(n), the number of primes <= n, at classical checkpoints.
    EXPECT_EQ(pi(10), 4u);
    EXPECT_EQ(pi(100), 25u);
    EXPECT_EQ(pi(1000), 168u);
    EXPECT_EQ(pi(10000), 1229u);
}

TEST(SieveOfEratosthenes, BoundaryIsInclusive) {
    // n itself is included when prime and excluded when composite.
    EXPECT_EQ(primes_up_to(29).back(), 29u); // 29 is prime -> present
    EXPECT_EQ(primes_up_to(30).back(), 29u); // 30 is composite -> last prime is still 29
    EXPECT_EQ(pi(2), 1u);                     // only 2
    EXPECT_EQ(pi(3), 2u);                     // 2, 3
}

TEST(SieveOfEratosthenes, SieveAndListAgree) {
    // Every index the sieve marks prime must appear in the list, in order, and vice versa.
    const std::uint64_t n = 5000;
    const auto sieve = prime_sieve(n);
    const auto primes = primes_up_to(n);
    ASSERT_EQ(sieve.size(), n + 1);

    std::vector<std::uint64_t> from_sieve;
    for (std::uint64_t i = 0; i < sieve.size(); ++i)
        if (sieve[i]) from_sieve.push_back(i);
    EXPECT_EQ(from_sieve, primes);

    EXPECT_FALSE(sieve[0]);
    EXPECT_FALSE(sieve[1]);
}

TEST(SieveOfEratosthenes, CrossCheckAgainstTrialDivision) {
    // For every i up to the bound, the sieve's verdict must match an independent primality test,
    // certifying both that every "prime" really is prime and every "composite" really has a divisor.
    const std::uint64_t n = 2000;
    const auto sieve = prime_sieve(n);
    for (std::uint64_t i = 0; i <= n; ++i)
        EXPECT_EQ(sieve[i], is_prime_trial(i)) << "disagreement at i = " << i;
}

TEST(SieveOfEratosthenes, BelowTwoIsEmpty) {
    EXPECT_TRUE(prime_sieve(0).empty());
    EXPECT_TRUE(prime_sieve(1).empty());
    EXPECT_TRUE(primes_up_to(0).empty());
    EXPECT_TRUE(primes_up_to(1).empty());
}
