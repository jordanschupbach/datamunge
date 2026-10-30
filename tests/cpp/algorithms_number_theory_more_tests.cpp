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

TEST(LucasTests, JacobiMatchesEulerCriterionForPrimes) {
    using datamunge::algorithms::jacobi_symbol;
    for (std::uint64_t p : {3u, 5u, 7u, 11u, 101u, 1009u})
        for (long long a = -20; a <= 20; ++a) {
            const std::uint64_t r = static_cast<std::uint64_t>(((a % static_cast<long long>(p)) + static_cast<long long>(p)) % static_cast<long long>(p));
            std::uint64_t e = 1, b = r, k = (p - 1) / 2;
            while (k) { if (k & 1) e = e * b % p; b = b * b % p; k >>= 1; }
            const int expect = r == 0 ? 0 : (e == 1 ? 1 : -1);
            EXPECT_EQ(jacobi_symbol(a, p), expect) << a << " " << p;
        }
    EXPECT_EQ(jacobi_symbol(2, 15), 1);  // (2/3)(2/5) = (-1)(-1)
    EXPECT_THROW(jacobi_symbol(1, 4), std::invalid_argument);
}

TEST(LucasTests, SequenceMatchesRecurrence) {
    using datamunge::algorithms::lucas_sequence_mod;
    const long long P = 3, Q = -5;
    const std::uint64_t n = 1000003;
    long long U0 = 0, U1 = 1, V0 = 2, V1 = P;  // exact while small, then reduce
    std::vector<std::uint64_t> U{0, 1}, V{2, 3};
    long long u0 = 0, u1 = 1, v0 = 2, v1 = 3;
    const long long N = static_cast<long long>(n);
    for (int k = 2; k <= 60; ++k) {
        const long long u2 = ((P * u1 - Q * u0) % N + N) % N, v2 = ((P * v1 - Q * v0) % N + N) % N;
        U.push_back(static_cast<std::uint64_t>(u2)); V.push_back(static_cast<std::uint64_t>(v2));
        u0 = u1; u1 = u2; v0 = v1; v1 = v2;
    }
    (void)U0; (void)U1; (void)V0; (void)V1;
    for (std::uint64_t k = 0; k <= 60; ++k) {
        const auto t = lucas_sequence_mod(k, P, Q, n);
        EXPECT_EQ(t.U, U[k]) << k;
        EXPECT_EQ(t.V, V[k]) << k;
    }
}

TEST(LucasTests, KnownPseudoprimes) {
    using namespace datamunge::algorithms;
    // OEIS A217120 (Lucas-Selfridge pseudoprimes) and A217255 (strong Lucas pseudoprimes).
    for (std::uint64_t n : {323u, 377u, 1159u, 1829u, 3827u, 5459u, 5777u, 9071u, 9179u, 10877u})
        EXPECT_TRUE(lucas_probable_prime(n)) << n;
    for (std::uint64_t n : {5459u, 5777u, 10877u, 16109u, 18971u, 22499u, 24569u, 25199u, 40309u, 58519u})
        EXPECT_TRUE(strong_lucas_probable_prime(n)) << n;
    EXPECT_FALSE(strong_lucas_probable_prime(323));
    EXPECT_FALSE(strong_lucas_probable_prime(49));  // square
    // Primes pass both; Baillie-PSW agrees with trial division below 200000.
    for (std::uint64_t n = 3; n < 200000; n += 2) {
        bool prime = true;
        for (std::uint64_t d = 3; d * d <= n; d += 2) if (n % d == 0) { prime = false; break; }
        if (prime) {
            ASSERT_TRUE(lucas_probable_prime(n)) << n;
            ASSERT_TRUE(strong_lucas_probable_prime(n)) << n;
        }
        ASSERT_EQ(baillie_psw(n), prime) << n;
    }
    // Large primes near 2^64.
    EXPECT_TRUE(baillie_psw(18446744073709551557ULL));
    EXPECT_TRUE(strong_lucas_probable_prime(18446744073709551557ULL));
    EXPECT_FALSE(baillie_psw(18446744073709551557ULL - 2));
    const auto s = selfridge_parameters(5459);
    EXPECT_EQ(s.Q, (1 - s.D) / 4);
    EXPECT_EQ(jacobi_symbol(s.D, 5459), -1);
}
