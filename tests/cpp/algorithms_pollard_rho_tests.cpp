#include <gtest/gtest.h>

#include <datamunge/algorithms/pollard_rho.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using datamunge::algorithms::factorize;
using datamunge::algorithms::pollard_rho_factor;

namespace {

// Independent primality check by trial division -- deliberately unrelated to the library's
// Miller-Rabin, so the tests never trust the implementation to grade itself.
bool is_prime_trial(std::uint64_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (std::uint64_t d = 3; d <= n / d; d += 2)
        if (n % d == 0) return false;
    return true;
}

std::uint64_t product(const std::vector<std::uint64_t>& v) {
    std::uint64_t p = 1;
    for (std::uint64_t x : v) p *= x;
    return p;
}

// Odd primes up to `limit`, by a simple sieve, for constructing controlled composites.
std::vector<std::uint64_t> small_primes(std::uint64_t limit) {
    std::vector<bool> sieve(limit + 1, true);
    std::vector<std::uint64_t> primes;
    for (std::uint64_t i = 2; i <= limit; ++i) {
        if (!sieve[i]) continue;
        primes.push_back(i);
        for (std::uint64_t j = i * i; j <= limit; j += i) sieve[j] = false;
    }
    return primes;
}

} // namespace

TEST(PollardRho, FactorizesSmallKnownNumbers) {
    EXPECT_EQ(factorize(12), (std::vector<std::uint64_t>{2, 2, 3}));
    EXPECT_EQ(factorize(1), (std::vector<std::uint64_t>{}));
    EXPECT_EQ(factorize(2), (std::vector<std::uint64_t>{2}));
    EXPECT_EQ(factorize(100), (std::vector<std::uint64_t>{2, 2, 5, 5}));
    EXPECT_EQ(factorize(97), (std::vector<std::uint64_t>{97}));                 // prime -> itself
    EXPECT_EQ(factorize(8051), (std::vector<std::uint64_t>{83, 97}));           // semiprime 83 * 97
    EXPECT_EQ(factorize(1024), std::vector<std::uint64_t>(10, 2));              // 2^10
}

TEST(PollardRho, FactorizesProjectEulerNumber) {
    // The classic 600851475143 = 71 * 839 * 1471 * 6857.
    EXPECT_EQ(factorize(600851475143ull), (std::vector<std::uint64_t>{71, 839, 1471, 6857}));
}

TEST(PollardRho, LargePrimeFactorsToItself) {
    const std::uint64_t p = 1000000007ull; // 10^9 + 7, prime
    ASSERT_TRUE(is_prime_trial(p));
    EXPECT_EQ(factorize(p), (std::vector<std::uint64_t>{p}));
    EXPECT_EQ(pollard_rho_factor(p), p); // documented: prime input returns n unchanged
}

TEST(PollardRho, FactorizesProductOfTwoLargePrimes) {
    // RSA-shaped: a product of two ~10-digit primes must be split, not just declared composite.
    const std::uint64_t p = 1000000007ull;  // 10^9 + 7
    const std::uint64_t q = 1000000009ull;  // 10^9 + 9
    ASSERT_TRUE(is_prime_trial(p));
    ASSERT_TRUE(is_prime_trial(q));
    const std::uint64_t n = p * q; // ~10^18, fits in uint64_t
    EXPECT_EQ(factorize(n), (std::vector<std::uint64_t>{p, q}));

    const std::uint64_t d = pollard_rho_factor(n);
    EXPECT_TRUE(d == p || d == q);
    EXPECT_EQ(n % d, 0u);
}

TEST(PollardRho, RejectsZero) {
    EXPECT_THROW((void)factorize(0), std::invalid_argument);
}

TEST(PollardRho, ProperDivisorForRandomSemiprimes) {
    // pollard_rho_factor must return a proper divisor 1 < d < n of each composite.
    const auto primes = small_primes(50000);
    std::mt19937_64 rng(20240927);
    std::uniform_int_distribution<std::size_t> pick(0, primes.size() - 1);

    for (int trial = 0; trial < 500; ++trial) {
        const std::uint64_t p = primes[pick(rng)];
        const std::uint64_t q = primes[pick(rng)];
        const std::uint64_t n = p * q;
        if (n < 4) continue;
        const std::uint64_t d = pollard_rho_factor(n);
        ASSERT_GT(d, 1u) << "n=" << n;
        ASSERT_LT(d, n) << "n=" << n;
        ASSERT_EQ(n % d, 0u) << "n=" << n << " d=" << d;
    }
}

TEST(PollardRho, RandomFactorizationsMultiplyBackAndArePrimeAndSorted) {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<std::uint64_t> dist(2, 1000000000000ull); // up to 10^12
    for (int trial = 0; trial < 300; ++trial) {
        const std::uint64_t n = dist(rng);
        const auto f = factorize(n);
        EXPECT_EQ(product(f), n) << "n=" << n;                                   // multiplies back
        EXPECT_TRUE(std::is_sorted(f.begin(), f.end())) << "n=" << n;            // ascending
        for (std::uint64_t p : f)
            EXPECT_TRUE(is_prime_trial(p)) << "n=" << n << " factor " << p;      // every factor prime
    }
}

TEST(PollardRho, EvenNumbersReturnTwo) {
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<std::uint64_t> dist(1, 1000000000ull);
    for (int trial = 0; trial < 100; ++trial) {
        const std::uint64_t n = 2 * dist(rng); // even, >= 2
        EXPECT_EQ(pollard_rho_factor(n), 2u) << "n=" << n;
    }
}
