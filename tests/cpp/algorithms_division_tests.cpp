#include <gtest/gtest.h>

#include <datamunge/algorithms/division.hpp>

#include <cstdint>
#include <random>

using namespace datamunge::algorithms;

namespace {

void check(std::uint64_t a, std::uint64_t b) {
    const auto r1 = restoring_divide(a, b);
    const auto r2 = non_restoring_divide(a, b);
    const auto r3 = newton_raphson_divide(a, b);
    EXPECT_EQ(r1.quotient, a / b) << "restoring " << a << "/" << b;
    EXPECT_EQ(r1.remainder, a % b) << "restoring " << a << "%" << b;
    EXPECT_EQ(r2.quotient, a / b) << "non-restoring " << a << "/" << b;
    EXPECT_EQ(r2.remainder, a % b) << "non-restoring " << a << "%" << b;
    EXPECT_EQ(r3.quotient, a / b) << "newton " << a << "/" << b;
    EXPECT_EQ(r3.remainder, a % b) << "newton " << a << "%" << b;
}

} // namespace

TEST(Division, RandomSmall) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 300000; ++t) {
        const std::uint64_t a = rng() % 100000000000ULL;
        std::uint64_t       b = rng() % 1000000ULL;
        if (b == 0) b = 1;
        check(a, b);
    }
}

TEST(Division, RandomFullWidth) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 200000; ++t) {
        const std::uint64_t a = rng();
        std::uint64_t       b = rng();
        if (b == 0) b = 1;
        check(a, b);
    }
}

TEST(Division, Corners) {
    check(100, 7);
    check(0, 5);
    check(5, 5);
    check(1, 1);
    check(4, 5);            // dividend < divisor
    check(1234567890, 1);   // divide by one
    check(0xFFFFFFFFFFFFFFFFULL, 3);
    check(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
    check(0xFFFFFFFFFFFFFFFFULL, 2);
    check(1ULL << 63, (1ULL << 62) + 1);
}

TEST(Division, KnownValues) {
    EXPECT_EQ(restoring_divide(100, 7).quotient, 14u);
    EXPECT_EQ(restoring_divide(100, 7).remainder, 2u);
    EXPECT_EQ(non_restoring_divide(100, 7).quotient, 14u);
    EXPECT_EQ(non_restoring_divide(100, 7).remainder, 2u);
    EXPECT_EQ(newton_raphson_divide(1000000, 999).quotient, 1001u);
    EXPECT_EQ(newton_raphson_divide(1000000, 999).remainder, 1u);
    // Division by zero is guarded, not UB.
    EXPECT_EQ(restoring_divide(5, 0).quotient, 0u);
    EXPECT_EQ(non_restoring_divide(5, 0).quotient, 0u);
    EXPECT_EQ(newton_raphson_divide(5, 0).quotient, 0u);
}
