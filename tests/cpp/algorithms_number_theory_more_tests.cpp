#include <gtest/gtest.h>

#include <datamunge/algorithms/number_theory_more.hpp>

#include <cstdint>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {
std::uint64_t powmod(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
    std::uint64_t r = 1 % m; a %= m;
    while (e) { if (e & 1) r = static_cast<std::uint64_t>((static_cast<__uint128_t>(r) * a) % m); a = static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * a) % m); e >>= 1; }
    return r;
}
bool trial_prime(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true;
}
} // namespace

TEST(NumberTheoryMore, PollardRhoLogRecoversExponent) {
    // Primes with a primitive root 3; g has order p-1.
    const std::uint64_t cases[][2] = {{1019, 2}, {2039, 7}, {10007, 5}, {104729, 6}};
    std::mt19937_64     rng(1);
    for (auto& c : cases) {
        const std::uint64_t p = c[0], g = c[1];
        const std::uint64_t order = p - 1;
        for (int t = 0; t < 20; ++t) {
            const std::uint64_t x = rng() % order;
            const std::uint64_t h = powmod(g, x, p);
            const long long     rx = pollard_rho_log(g, h, p, order);
            ASSERT_GE(rx, 0) << "p=" << p << " x=" << x;
            EXPECT_EQ(powmod(g, static_cast<std::uint64_t>(rx), p), h) << "p=" << p;
        }
    }
}

TEST(NumberTheoryMore, PohligHellmanRecoversExponent) {
    // Primes whose p-1 is smooth, so Pohlig-Hellman is the natural choice.
    const std::uint64_t cases[][2] = {{1009, 11}, {2003, 5}, {65537, 3}, {1000003, 2}};
    std::mt19937_64     rng(2);
    for (auto& c : cases) {
        const std::uint64_t p = c[0], g = c[1];
        const std::uint64_t order = p - 1;
        for (int t = 0; t < 30; ++t) {
            const std::uint64_t x = rng() % order;
            const std::uint64_t h = powmod(g, x, p);
            const long long     rx = pohlig_hellman_log(g, h, p, order);
            ASSERT_GE(rx, 0) << "p=" << p << " x=" << x;
            EXPECT_EQ(powmod(g, static_cast<std::uint64_t>(rx), p), h) << "p=" << p << " x=" << x;
        }
    }
}

TEST(NumberTheoryMore, BailliePswMatchesTrialDivision) {
    for (std::uint64_t n = 0; n < 200000; ++n) EXPECT_EQ(baillie_psw(n), trial_prime(n)) << "n=" << n;
    // A few large primes and composites.
    EXPECT_TRUE(baillie_psw(1000000007ULL));
    EXPECT_TRUE(baillie_psw(9223372036854775783ULL)); // largest prime < 2^63
    EXPECT_FALSE(baillie_psw(1000000007ULL * 3ULL));
    EXPECT_FALSE(baillie_psw(3215031751ULL));          // a strong pseudoprime to bases 2,3,5,7 (must fail Lucas)
    EXPECT_FALSE(baillie_psw(25326001ULL));            // spsp(2,3,5)
}
