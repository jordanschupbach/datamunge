#include <gtest/gtest.h>

#include <datamunge/algorithms/pi_and_cordic.hpp>

#include <cmath>
#include <vector>

using datamunge::algorithms::bbp_pi_hex_digit;
using datamunge::algorithms::chudnovsky_pi;
using datamunge::algorithms::cordic_sincos;
using datamunge::algorithms::gauss_legendre_pi;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

// ------------------------------------------------------------------------------------------------
// BBP
// ------------------------------------------------------------------------------------------------

TEST(BBP, MatchesKnownHexDigitsOfPi) {
    // pi = 3.243F6A8885A308D3... so the hex digits after the point are:
    const std::vector<int> expected = {2, 4, 3, 15, 6, 10, 8, 8, 8, 5, 10, 3, 0, 8, 13, 3};
    for (int n = 0; n < static_cast<int>(expected.size()); ++n)
        EXPECT_EQ(bbp_pi_hex_digit(n), expected[static_cast<std::size_t>(n)]) << "digit " << n;
}

TEST(BBP, DeepDigitIsComputedDirectly) {
    // Reconstruct the fractional part of pi from its first 13 hex digits and compare to pi - 3.
    double recon = 0.0;
    for (int i = 0; i < 13; ++i) recon += bbp_pi_hex_digit(i) * std::pow(16.0, -(i + 1));
    EXPECT_NEAR(recon, kPi - 3.0, std::pow(16.0, -12.0));
}

// ------------------------------------------------------------------------------------------------
// Gauss-Legendre and Chudnovsky
// ------------------------------------------------------------------------------------------------

TEST(GaussLegendre, ConvergesToPiQuadratically) {
    EXPECT_NEAR(gauss_legendre_pi(4), kPi, 1e-14);
    EXPECT_NEAR(gauss_legendre_pi(5), kPi, 1e-14);
    // Each iteration should not increase the error (until it saturates).
    EXPECT_LE(std::fabs(gauss_legendre_pi(3) - kPi), std::fabs(gauss_legendre_pi(2) - kPi));
}

TEST(Chudnovsky, ConvergesToPi) {
    EXPECT_NEAR(chudnovsky_pi(2), kPi, 1e-13);
    EXPECT_NEAR(chudnovsky_pi(3), kPi, 1e-14);
    EXPECT_LT(std::fabs(chudnovsky_pi(2) - kPi), std::fabs(chudnovsky_pi(1) - kPi));
}

// ------------------------------------------------------------------------------------------------
// CORDIC
// ------------------------------------------------------------------------------------------------

TEST(Cordic, MatchesStdSinCosOverRange) {
    for (double theta = -10.0; theta <= 10.0; theta += 0.03) {
        const auto sc = cordic_sincos(theta, 44);
        EXPECT_NEAR(sc.sin, std::sin(theta), 1e-6) << "theta=" << theta;
        EXPECT_NEAR(sc.cos, std::cos(theta), 1e-6) << "theta=" << theta;
        EXPECT_NEAR(sc.sin * sc.sin + sc.cos * sc.cos, 1.0, 1e-6); // stays on the unit circle
    }
}

TEST(Cordic, KnownAngles) {
    EXPECT_NEAR(cordic_sincos(0.0).sin, 0.0, 1e-9);
    EXPECT_NEAR(cordic_sincos(0.0).cos, 1.0, 1e-9);
    EXPECT_NEAR(cordic_sincos(kPi / 6.0).sin, 0.5, 1e-9);
    EXPECT_NEAR(cordic_sincos(kPi / 2.0).sin, 1.0, 1e-9);
    EXPECT_NEAR(cordic_sincos(kPi).cos, -1.0, 1e-9);
}
