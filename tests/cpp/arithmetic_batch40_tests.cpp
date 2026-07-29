#include <gtest/gtest.h>

#include <datamunge/algorithms/borwein_pi.hpp>
#include <datamunge/algorithms/rounding_functions.hpp>
#include <datamunge/algorithms/srt_division.hpp>

#include <cfenv>
#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Borwein quartic 1/pi ----------------

TEST(BorweinPi, ConvergesToPi) {
    // Double precision saturates a few ulp from pi (~1.3e-15).
    EXPECT_NEAR(borwein_pi(3), M_PI, 3e-15);
    EXPECT_NEAR(borwein_pi(4), M_PI, 3e-15);
}

TEST(BorweinPi, QuarticErrorCollapse) {
    const auto seq = borwein_pi_sequence(3);
    ASSERT_EQ(seq.size(), 4u);
    // Errors must fall monotonically and dramatically each step (quartic).
    double e0 = std::fabs(seq[0] - M_PI);
    double e1 = std::fabs(seq[1] - M_PI);
    double e2 = std::fabs(seq[2] - M_PI);
    EXPECT_LT(e1, e0);
    EXPECT_LT(e2, e1);
    EXPECT_LT(e1, 1e-3);   // one step already many digits
    EXPECT_LT(e2, 1e-14);  // two steps saturate double
}

// ---------------- SRT division ----------------

TEST(SrtDivision, MatchesTrueQuotient) {
    std::mt19937_64                        rng(9);
    std::uniform_real_distribution<double> dd(0.01, 1000.0);
    for (int t = 0; t < 5000; ++t) {
        const double D = dd(rng);
        const double N = std::uniform_real_distribution<double>(0.0, D)(rng); // 0 <= N < D
        const double q = srt_divide(N, D, 52);
        EXPECT_NEAR(q, N / D, 1e-12) << "N=" << N << " D=" << D;
    }
}

TEST(SrtDivision, RedundantDigitsInRange) {
    const auto digits = srt_quotient_digits(0.3, 0.8, 30);
    for (int q : digits) EXPECT_TRUE(q == -1 || q == 0 || q == 1);
    EXPECT_NEAR(srt_digits_to_value(digits), 0.3 / 0.8, 1e-8);
}

TEST(SrtDivision, PrecisionScalesWithBits) {
    const double N = 0.375, D = 0.6;
    EXPECT_NEAR(srt_divide(N, D, 10), N / D, 1e-2);
    EXPECT_NEAR(srt_divide(N, D, 20), N / D, 1e-5);
    EXPECT_NEAR(srt_divide(N, D, 40), N / D, 1e-11);
}

// ---------------- Rounding functions ----------------

TEST(Rounding, TieBreakingCases) {
    EXPECT_EQ(round_half_to_even(0.5), 0.0);
    EXPECT_EQ(round_half_to_even(1.5), 2.0);
    EXPECT_EQ(round_half_to_even(2.5), 2.0);
    EXPECT_EQ(round_half_to_even(-0.5), 0.0);
    EXPECT_EQ(round_half_to_even(-1.5), -2.0);

    EXPECT_EQ(round_half_away_from_zero(0.5), 1.0);
    EXPECT_EQ(round_half_away_from_zero(2.5), 3.0);
    EXPECT_EQ(round_half_away_from_zero(-0.5), -1.0);

    EXPECT_EQ(round_half_up(0.5), 1.0);
    EXPECT_EQ(round_half_up(-0.5), 0.0);
    EXPECT_EQ(round_half_down(0.5), 0.0);
    EXPECT_EQ(round_half_down(-0.5), -1.0);
    EXPECT_EQ(round_toward_zero(2.9), 2.0);
    EXPECT_EQ(round_toward_zero(-2.9), -2.0);
}

TEST(Rounding, HalfToEvenMatchesNearbyint) {
    std::fesetround(FE_TONEAREST);
    std::mt19937_64                        rng(3);
    std::uniform_real_distribution<double> d(-1000, 1000);
    for (int t = 0; t < 10000; ++t) {
        const double x = d(rng);
        EXPECT_EQ(round_half_to_even(x), std::nearbyint(x)) << "x=" << x;
    }
    // and exact half-integers, where nearbyint uses round-half-to-even too
    for (int k = -5; k <= 5; ++k)
        EXPECT_EQ(round_half_to_even(k + 0.5), std::nearbyint(k + 0.5)) << "k=" << k;
}

TEST(Rounding, HalfToEvenIsUnbiasedHalfUpIsNot) {
    // Tie-heavy data: all the half-integers k + 0.5.
    std::vector<double> vals;
    for (int k = -50; k <= 49; ++k) vals.push_back(k + 0.5);
    const double even_bias = rounding_bias(vals, RoundMode::HalfToEven);
    const double up_bias   = rounding_bias(vals, RoundMode::HalfUp);
    EXPECT_NEAR(even_bias, 0.0, 1e-12);
    EXPECT_NEAR(up_bias, 0.5, 1e-12); // every tie rounds up by 0.5
}
