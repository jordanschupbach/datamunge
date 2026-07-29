#include <gtest/gtest.h>

#include <datamunge/algorithms/chinese_remainder.hpp>

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using datamunge::algorithms::chinese_remainder;
using datamunge::algorithms::CrtSolution;

namespace {

// Euclidean helpers for cross-checking, independent of the implementation under test.
std::int64_t igcd(std::int64_t a, std::int64_t b) { return b == 0 ? a : igcd(b, a % b); }
std::int64_t mod_pos(std::int64_t a, std::int64_t m) { return ((a % m) + m) % m; }

} // namespace

TEST(ChineseRemainder, SunTzuClassicProblem) {
    // Sun-Tzu (3rd century): x = 2 (mod 3), 3 (mod 5), 2 (mod 7)  ->  23 (mod 105).
    const auto s = chinese_remainder({2, 3, 2}, {3, 5, 7});
    EXPECT_TRUE(s.solvable);
    EXPECT_EQ(s.remainder, 23);
    EXPECT_EQ(s.modulus, 105);
}

TEST(ChineseRemainder, CoprimeModuliMatchProduct) {
    // Coprime moduli: the merged modulus is the product 3*4*5 = 60.
    const auto s = chinese_remainder({1, 2, 3}, {3, 4, 5});
    EXPECT_TRUE(s.solvable);
    EXPECT_EQ(s.modulus, 60);
    EXPECT_EQ(mod_pos(s.remainder, 3), 1);
    EXPECT_EQ(mod_pos(s.remainder, 4), 2);
    EXPECT_EQ(mod_pos(s.remainder, 5), 3);
    EXPECT_GE(s.remainder, 0);
    EXPECT_LT(s.remainder, s.modulus);
}

TEST(ChineseRemainder, NonCoprimeButConsistent) {
    // x = 2 (mod 6), x = 8 (mod 12): gcd(6,12)=6 divides (2-8)=-6, so consistent.
    // Solution is 8 (mod lcm(6,12)=12).
    const auto s = chinese_remainder({2, 8}, {6, 12});
    EXPECT_TRUE(s.solvable);
    EXPECT_EQ(s.modulus, 12);
    EXPECT_EQ(s.remainder, 8);
    EXPECT_EQ(mod_pos(s.remainder, 6), 2);
    EXPECT_EQ(mod_pos(s.remainder, 12), 8);
}

TEST(ChineseRemainder, InconsistentSystemIsUnsolvable) {
    // x = 1 (mod 2) forces x odd; x = 0 (mod 4) forces x even. gcd(2,4)=2 does NOT divide (1-0)=1.
    const auto s = chinese_remainder({1, 0}, {2, 4});
    EXPECT_FALSE(s.solvable);
}

TEST(ChineseRemainder, SingleCongruenceNormalizesResidue) {
    // A single congruence is returned verbatim, with a negative residue normalized into [0, m).
    const auto s = chinese_remainder({-3}, {7});
    EXPECT_TRUE(s.solvable);
    EXPECT_EQ(s.modulus, 7);
    EXPECT_EQ(s.remainder, 4); // -3 == 4 (mod 7)
}

TEST(ChineseRemainder, EmptySystemIsEveryInteger) {
    // The empty system is vacuously true: x == 0 (mod 1).
    const auto s = chinese_remainder({}, {});
    EXPECT_TRUE(s.solvable);
    EXPECT_EQ(s.remainder, 0);
    EXPECT_EQ(s.modulus, 1);
}

TEST(ChineseRemainder, NegativeRemaindersAreNormalized) {
    // Negative targets must be reduced into [0, m) before merging.
    const auto s = chinese_remainder({-1, -1}, {3, 5});
    EXPECT_TRUE(s.solvable);
    EXPECT_EQ(s.modulus, 15);
    EXPECT_EQ(s.remainder, 14); // -1 == 14 (mod 15)
}

TEST(ChineseRemainder, RandomizedRoundTripCrossCheck) {
    // Pick a random x and random moduli, form the true residues, solve, and assert the recovered
    // solution agrees with x modulo every input modulus AND modulo their lcm.
    std::mt19937_64 rng(20260727);
    for (int trial = 0; trial < 5000; ++trial) {
        const int k = 1 + static_cast<int>(rng() % 5);
        const std::int64_t x = static_cast<std::int64_t>(rng() % 4000001) - 2000000;
        std::vector<std::int64_t> moduli, remainders;
        for (int i = 0; i < k; ++i) {
            const std::int64_t m = 1 + static_cast<std::int64_t>(rng() % 1000);
            moduli.push_back(m);
            remainders.push_back(mod_pos(x, m));
        }
        const auto s = chinese_remainder(remainders, moduli);
        ASSERT_TRUE(s.solvable) << "trial " << trial; // constructed from a real x, must be solvable

        // lcm of all moduli, computed independently.
        std::int64_t lcm = 1;
        for (const std::int64_t m : moduli) lcm = lcm / igcd(lcm, m) * m;
        EXPECT_EQ(s.modulus, lcm) << "trial " << trial;

        EXPECT_GE(s.remainder, 0);
        EXPECT_LT(s.remainder, s.modulus);
        for (const std::int64_t m : moduli)
            EXPECT_EQ(mod_pos(s.remainder, m), mod_pos(x, m)) << "trial " << trial;
        EXPECT_EQ(mod_pos(x - s.remainder, lcm), 0) << "trial " << trial; // x == solution (mod lcm)
    }
}

TEST(ChineseRemainder, RandomizedInconsistencyIsDetected) {
    // Two congruences with a deliberately impossible offset must be reported unsolvable exactly
    // when gcd(m1,m2) fails to divide the residue difference.
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 3000; ++trial) {
        const std::int64_t m1 = 2 + static_cast<std::int64_t>(rng() % 60);
        const std::int64_t m2 = 2 + static_cast<std::int64_t>(rng() % 60);
        const std::int64_t r1 = static_cast<std::int64_t>(rng() % m1);
        const std::int64_t r2 = static_cast<std::int64_t>(rng() % m2);
        const auto s = chinese_remainder({r1, r2}, {m1, m2});
        const bool should_solve = (mod_pos(r1 - r2, igcd(m1, m2)) == 0);
        EXPECT_EQ(s.solvable, should_solve) << "trial " << trial;
    }
}

TEST(ChineseRemainder, RejectsMalformedInput) {
    EXPECT_THROW((void)chinese_remainder({1, 2}, {3}), std::invalid_argument);    // length mismatch
    EXPECT_THROW((void)chinese_remainder({1}, {0}), std::invalid_argument);       // zero modulus
    EXPECT_THROW((void)chinese_remainder({1}, {-5}), std::invalid_argument);      // negative modulus
}
