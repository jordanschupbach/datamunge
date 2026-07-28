#include <gtest/gtest.h>

#include <datamunge/algorithms/elementary_more.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>

using namespace datamunge::algorithms;

TEST(ElementaryMore, BinarySplittingApproximatesE) {
    const double e = std::exp(1.0);
    double prev = 1.0;
    for (int terms = 4; terms <= 18; terms += 2) {
        const auto r = binary_splitting_e(terms);
        EXPECT_LT(std::fabs(r.value - e), prev); // converges
        prev = std::fabs(r.value - e);
        // P/Q as a double equals the reported value.
        EXPECT_DOUBLE_EQ(r.value, static_cast<double>(r.num) / static_cast<double>(r.den));
    }
    EXPECT_NEAR(binary_splitting_e(18).value, e, 1e-12);
    // 3 terms: 1 + 1 + 1/2 = 5/2.
    const auto three = binary_splitting_e(3);
    EXPECT_EQ(three.num, 5u);
    EXPECT_EQ(three.den, 2u);
}

TEST(ElementaryMore, SpigotProducesEDigits) {
    // e = 2.718281828459045235360287471352|662497757...
    EXPECT_EQ(spigot_e(30), "2.718281828459045235360287471352");
    EXPECT_EQ(spigot_e(10), "2.7182818284");
    EXPECT_EQ(spigot_e(50).substr(0, 22), "2.71828182845904523536");
}

TEST(ElementaryMore, DigitByDigitSqrtMatchesStd) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 2000; ++t) {
        const std::uint64_t n = rng() % 1000000ULL;
        const std::string   s = digit_by_digit_sqrt(n, 8);
        const double        got = std::stod(s);
        EXPECT_NEAR(got, std::sqrt(static_cast<double>(n)), 1e-6) << "n=" << n << " s=" << s;
    }
    // Known expansions.
    EXPECT_EQ(digit_by_digit_sqrt(2, 10).substr(0, 12), "1.4142135623");
    EXPECT_EQ(digit_by_digit_sqrt(152.0 ? 152 : 0, 4).substr(0, 2), "12"); // sqrt(152)=12.3288...
    EXPECT_EQ(digit_by_digit_sqrt(144, 2), "12.00"); // perfect square
    EXPECT_EQ(digit_by_digit_sqrt(0, 3), "0.000");
}

TEST(ElementaryMore, LongDivisionMatchesBuiltin) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 500000; ++t) {
        const std::uint64_t a = rng() % 100000000000ULL;
        std::uint64_t       b = rng() % 1000000ULL;
        if (b == 0) b = 1;
        const auto r = long_division(a, b);
        EXPECT_EQ(r.quotient, a / b);
        EXPECT_EQ(r.remainder, a % b);
        EXPECT_EQ(r.quotient * b + r.remainder, a);
    }
    EXPECT_EQ(long_division(100, 7).quotient, 14u);
    EXPECT_EQ(long_division(100, 7).remainder, 2u);
}
