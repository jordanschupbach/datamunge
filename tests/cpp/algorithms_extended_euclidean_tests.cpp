#include <gtest/gtest.h>

#include <datamunge/algorithms/extended_euclidean.hpp>

#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>

using datamunge::algorithms::extended_gcd;
using datamunge::algorithms::ExtendedGcd;
using datamunge::algorithms::modular_inverse;

TEST(ExtendedEuclidean, KnownGcdAndBezout) {
    const ExtendedGcd g = extended_gcd(240, 46);
    EXPECT_EQ(g.gcd, 2);
    // The gcd(240, 46) = 2, and the Bezout identity must hold with the returned coefficients.
    EXPECT_EQ(240 * g.x + 46 * g.y, g.gcd);
    // The canonical iterative solution for this pair is x = -9, y = 47.
    EXPECT_EQ(g.x, -9);
    EXPECT_EQ(g.y, 47);
}

TEST(ExtendedEuclidean, BezoutIdentityHoldsOnRandomPairs) {
    std::mt19937_64 rng(2024);
    std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
    for (int trial = 0; trial < 5000; ++trial) {
        const std::int64_t a = dist(rng), b = dist(rng);
        const ExtendedGcd g = extended_gcd(a, b);
        EXPECT_EQ(a * g.x + b * g.y, g.gcd) << "a=" << a << " b=" << b;
    }
}

TEST(ExtendedEuclidean, GcdMatchesStdGcdAndIsNonNegative) {
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
    for (int trial = 0; trial < 5000; ++trial) {
        const std::int64_t a = dist(rng), b = dist(rng);
        const ExtendedGcd g = extended_gcd(a, b);
        EXPECT_GE(g.gcd, 0);
        EXPECT_EQ(g.gcd, static_cast<std::int64_t>(std::gcd(a, b))) << "a=" << a << " b=" << b;
    }
}

TEST(ExtendedEuclidean, ModularInverseIsCorrectAndCanonical) {
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<std::int64_t> mod_dist(2, 100'000);
    std::uniform_int_distribution<std::int64_t> a_dist(-1'000'000, 1'000'000);
    int tested = 0;
    while (tested < 5000) {
        const std::int64_t m = mod_dist(rng);
        const std::int64_t a = a_dist(rng);
        if (std::gcd(a, m) != 1) continue; // only coprime pairs are invertible
        const std::int64_t inv = modular_inverse(a, m);
        EXPECT_GE(inv, 0);
        EXPECT_LT(inv, m);
        // a * inv ≡ 1 (mod m); reduce a into range first to keep the product small and positive.
        std::int64_t ar = a % m;
        if (ar < 0) ar += m;
        EXPECT_EQ((ar * inv) % m, 1) << "a=" << a << " m=" << m << " inv=" << inv;
        ++tested;
    }
}

TEST(ExtendedEuclidean, ModularInverseKnownValues) {
    EXPECT_EQ(modular_inverse(3, 11), 4);   // 3 * 4 = 12 ≡ 1 (mod 11)
    EXPECT_EQ(modular_inverse(7, 26), 15);  // 7 * 15 = 105 ≡ 1 (mod 26)
    EXPECT_EQ(modular_inverse(-3, 11), 7);  // -3 ≡ 8 (mod 11); 8 * 7 = 56 ≡ 1 (mod 11)
    EXPECT_EQ(modular_inverse(1, 7), 1);
}

TEST(ExtendedEuclidean, ThrowsWhenNotInvertible) {
    EXPECT_THROW((void)modular_inverse(6, 9), std::invalid_argument);   // gcd(6, 9) = 3
    EXPECT_THROW((void)modular_inverse(4, 8), std::invalid_argument);   // gcd(4, 8) = 4
    EXPECT_THROW((void)modular_inverse(0, 5), std::invalid_argument);   // gcd(0, 5) = 5
}

TEST(ExtendedEuclidean, ThrowsOnNonPositiveModulus) {
    EXPECT_THROW((void)modular_inverse(3, 0), std::invalid_argument);
    EXPECT_THROW((void)modular_inverse(3, -11), std::invalid_argument);
}

TEST(ExtendedEuclidean, EdgeCasesZeroAndNegatives) {
    // gcd(0, 0) = 0 with trivial coefficients.
    const ExtendedGcd z = extended_gcd(0, 0);
    EXPECT_EQ(z.gcd, 0);
    EXPECT_EQ(0 * z.x + 0 * z.y, 0);

    // gcd(0, b) = |b|.
    const ExtendedGcd a0 = extended_gcd(0, -12);
    EXPECT_EQ(a0.gcd, 12);
    EXPECT_EQ(0 * a0.x + (-12) * a0.y, a0.gcd);

    // Negative inputs still satisfy the identity and yield a non-negative gcd.
    const ExtendedGcd n = extended_gcd(-240, 46);
    EXPECT_EQ(n.gcd, 2);
    EXPECT_EQ(-240 * n.x + 46 * n.y, n.gcd);

    const ExtendedGcd n2 = extended_gcd(-240, -46);
    EXPECT_EQ(n2.gcd, 2);
    EXPECT_EQ(-240 * n2.x + -46 * n2.y, n2.gcd);
}
