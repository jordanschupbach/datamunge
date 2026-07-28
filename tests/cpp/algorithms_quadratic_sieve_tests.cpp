#include <gtest/gtest.h>

#include <datamunge/algorithms/quadratic_sieve.hpp>

#include <cstdint>
#include <vector>

using namespace datamunge::algorithms;

TEST(QuadraticSieve, FactorsSemiprimes) {
    const std::vector<std::uint64_t> ps = {10007, 10009, 100003, 1000003, 15485863, 32452843};
    const std::vector<std::uint64_t> qs = {10037, 20011, 200003, 1000033, 15485867, 49979687};
    for (std::uint64_t p : ps)
        for (std::uint64_t q : qs) {
            const std::uint64_t n = p * q;
            const std::uint64_t f = quadratic_sieve(n);
            EXPECT_GT(f, 1u) << n;
            EXPECT_LT(f, n) << n;
            EXPECT_EQ(n % f, 0u) << "n=" << n << " f=" << f;
        }
}

TEST(QuadraticSieve, SmallFactorsAndSquares) {
    EXPECT_EQ(quadratic_sieve(2ull * 7919), 2u);
    EXPECT_EQ(quadratic_sieve(3ull * 1000003), 3u);
    EXPECT_EQ(quadratic_sieve(9409), 97u);     // 97^2
    EXPECT_EQ(quadratic_sieve(1000003ull * 1000003), 1000003u); // prime square
    // A product of a small and a large prime.
    const std::uint64_t n = 101ull * 1299709;
    const std::uint64_t f = quadratic_sieve(n);
    EXPECT_EQ(n % f, 0u);
    EXPECT_GT(f, 1u);
    EXPECT_LT(f, n);
}

TEST(QuadraticSieve, HarderSemiprime) {
    // Two ~7-digit primes.
    const std::uint64_t n = 4000037ull * 7999993ull;
    const std::uint64_t f = quadratic_sieve(n);
    EXPECT_EQ(n % f, 0u) << "n=" << n << " f=" << f;
    EXPECT_GT(f, 1u);
    EXPECT_LT(f, n);
}
