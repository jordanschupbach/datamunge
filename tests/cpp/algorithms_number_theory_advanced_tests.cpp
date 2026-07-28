#include <gtest/gtest.h>

#include <datamunge/algorithms/number_theory_advanced.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace datamunge::algorithms;

namespace {

bool is_square(std::uint64_t n) {
    std::uint64_t r = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(n)));
    while (r * r > n) --r;
    while ((r + 1) * (r + 1) <= n) ++r;
    return r * r == n;
}

// Simple deterministic primality for cross-checking (trial division).
bool prime_ref(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t p = 2; p * p <= n; ++p)
        if (n % p == 0) return false;
    return true;
}

} // namespace

TEST(NumberTheoryAdvanced, ChakravalaSolvesPell) {
    for (std::uint64_t N = 2; N <= 150; ++N) {
        if (is_square(N)) continue;
        const auto sol = chakravala(N);
        ASSERT_TRUE(sol.found) << "N=" << N;
        // x^2 - N y^2 == 1, with y > 0.
        const __int128 lhs = sol.x * sol.x - static_cast<__int128>(N) * sol.y * sol.y;
        EXPECT_EQ(lhs, static_cast<__int128>(1)) << "N=" << N;
        EXPECT_GT(sol.y, 0) << "N=" << N;
    }
}

TEST(NumberTheoryAdvanced, ChakravalaKnownFundamental) {
    auto s2 = chakravala(2);
    EXPECT_EQ((long long)s2.x, 3);
    EXPECT_EQ((long long)s2.y, 2);
    auto s13 = chakravala(13);
    EXPECT_EQ((long long)s13.x, 649);
    EXPECT_EQ((long long)s13.y, 180);
    auto s61 = chakravala(61); // famously large fundamental solution
    EXPECT_EQ((long long)s61.x, 1766319049LL);
    EXPECT_EQ((long long)s61.y, 226153980LL);
    EXPECT_FALSE(chakravala(49).found); // perfect square
}

TEST(NumberTheoryAdvanced, LenstraEcmFindsFactor) {
    // Products of two primes; ECM must return a nontrivial divisor.
    const std::vector<std::uint64_t> composites = {
        1517,             // 37 * 41
        10403,            // 101 * 103
        104729ull * 1009, // two primes
        1299709ull * 101, // two primes
        997ull * 997,     // prime squared
        12345678ull,      // even
        999983ull * 3,    // multiple of 3
    };
    for (std::uint64_t n : composites) {
        const std::uint64_t f = lenstra_ecm(n);
        EXPECT_GT(f, 1u) << "n=" << n;
        EXPECT_LT(f, n) << "n=" << n;
        EXPECT_EQ(n % f, 0u) << "n=" << n << " f=" << f;
    }
}

TEST(NumberTheoryAdvanced, LenstraEcmRandomSemiprimes) {
    // Deterministic list of semiprimes p*q with moderate factors.
    const std::uint64_t ps[] = {101, 211, 307, 401, 503, 601, 701, 809, 907, 1009};
    const std::uint64_t qs[] = {1013, 1201, 1301, 1409, 1511, 1601, 1709, 1801, 1901, 2003};
    for (std::uint64_t p : ps)
        for (std::uint64_t q : qs) {
            const std::uint64_t n = p * q;
            const std::uint64_t f = lenstra_ecm(n);
            EXPECT_GT(f, 1u) << n;
            EXPECT_LT(f, n) << n;
            EXPECT_EQ(n % f, 0u) << "n=" << n << " f=" << f;
        }
}

TEST(NumberTheoryAdvanced, AksMatchesTrialDivision) {
    for (std::uint64_t n = 2; n <= 400; ++n)
        EXPECT_EQ(aks_is_prime(n), prime_ref(n)) << "n=" << n;
}

TEST(NumberTheoryAdvanced, AksKnownPrimes) {
    for (std::uint64_t p : {2ull, 3ull, 5ull, 31ull, 97ull, 541ull, 1009ull, 7919ull})
        EXPECT_TRUE(aks_is_prime(p)) << p;
    for (std::uint64_t c : {4ull, 9ull, 15ull, 91ull, 561ull, 1024ull, 7917ull})
        EXPECT_FALSE(aks_is_prime(c)) << c;
}
