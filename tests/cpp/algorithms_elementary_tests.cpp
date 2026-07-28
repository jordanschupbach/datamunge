#include <gtest/gtest.h>

#include <datamunge/algorithms/elementary.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

TEST(Elementary, KahanBeatsNaiveOnIllConditionedSum) {
    // A large value plus many tiny ones: naive summation loses the tiny contributions.
    std::vector<double> xs;
    xs.push_back(1.0e16);
    for (int i = 0; i < 1000000; ++i) xs.push_back(1.0);
    xs.push_back(-1.0e16);
    const double exact = 1000000.0;

    double naive = 0.0;
    for (double x : xs) naive += x;

    const double kahan = kahan_sum(xs);
    EXPECT_NEAR(kahan, exact, 1e-3);
    // Naive is far worse (it drops most of the 1.0s while the 1e16 dominates).
    EXPECT_GT(std::fabs(naive - exact), 1.0);
    EXPECT_LT(std::fabs(kahan - exact), std::fabs(naive - exact));
}

TEST(Elementary, KahanMatchesLongDoubleReference) {
    std::mt19937                          rng(7);
    std::uniform_real_distribution<double> d(-1e6, 1e6);
    std::vector<double>                    xs(100000);
    long double                            ref = 0.0L;
    for (double& x : xs) { x = d(rng); ref += static_cast<long double>(x); }
    EXPECT_NEAR(kahan_sum(xs), static_cast<double>(ref), 1e-4);
}

TEST(Elementary, NewtonDivisionMatchesTrueQuotient) {
    std::mt19937                          rng(11);
    std::uniform_real_distribution<double> d(-1e6, 1e6);
    for (int i = 0; i < 100000; ++i) {
        const double n = d(rng);
        double       den = d(rng);
        if (den == 0.0) den = 1.0;
        const double q = newton_raphson_division(n, den);
        EXPECT_NEAR(q, n / den, std::fabs(n / den) * 1e-12 + 1e-12);
    }
    // Specific tricky magnitudes.
    EXPECT_NEAR(newton_raphson_division(1.0, 3.0), 1.0 / 3.0, 1e-15);
    EXPECT_NEAR(newton_raphson_division(-7.0, 1024.0), -7.0 / 1024.0, 1e-15);
    EXPECT_NEAR(newton_raphson_division(5.0, 1e-9), 5.0 / 1e-9, std::fabs(5.0 / 1e-9) * 1e-12);
}

TEST(Elementary, NthRootMatchesStdPow) {
    std::mt19937                          rng(13);
    std::uniform_real_distribution<double> a(1e-6, 1e6);
    for (int i = 0; i < 50000; ++i) {
        const double v = a(rng);
        for (int n : {2, 3, 4, 5, 7, 10}) EXPECT_NEAR(nth_root(v, n), std::pow(v, 1.0 / n), std::pow(v, 1.0 / n) * 1e-10);
    }
    EXPECT_NEAR(nth_root(2.0, 2), std::sqrt(2.0), 1e-14);
    EXPECT_NEAR(nth_root(27.0, 3), 3.0, 1e-12);
    EXPECT_NEAR(nth_root(1024.0, 10), 2.0, 1e-12);
    EXPECT_EQ(nth_root(0.0, 4), 0.0);
    EXPECT_EQ(nth_root(5.0, 1), 5.0);
}

TEST(Elementary, AlphaMaxBetaMinWithinErrorBound) {
    std::mt19937                          rng(17);
    std::uniform_real_distribution<double> d(-100.0, 100.0);
    double                                 worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        const double x = d(rng), y = d(rng);
        const double exact = std::hypot(x, y);
        if (exact < 1e-9) continue;
        const double approx = alpha_max_beta_min(x, y);
        worst = std::max(worst, std::fabs(approx - exact) / exact);
    }
    // Peak relative error of the optimal coefficients is ~3.96%.
    EXPECT_LT(worst, 0.041);
    EXPECT_GT(worst, 0.02); // it really is an approximation, not exact
    // On axes it is exact (min component is 0 -> alpha*|max|, alpha~0.96 gives ~4% low; still within bound)
    EXPECT_NEAR(alpha_max_beta_min(3.0, 4.0) / 5.0, 1.0, 0.041);
}
