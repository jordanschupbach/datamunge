#include <datamunge/stats/polychoric.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <tuple>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_polychoric_data.inc"

std::vector<std::vector<double>> table(const std::vector<double>& v, std::size_t cols) {
    std::vector<std::vector<double>> t(v.size() / cols, std::vector<double>(cols));
    for (std::size_t i = 0; i < v.size(); ++i)
        t[i / cols][i % cols] = v[i];
    return t;
}

std::vector<int> ints(const std::vector<double>& v) { return {v.begin(), v.end()}; }
} // namespace

TEST(Polychoric, BivariateNormalCdf) {
    for (std::size_t i = 0; i < pc_bvn.size(); ++i)
        EXPECT_NEAR(bivariate_normal_cdf(pc_bvn_in[3 * i], pc_bvn_in[3 * i + 1], pc_bvn_in[3 * i + 2]), pc_bvn[i],
                    1e-12)
            << i;
    EXPECT_NEAR(bivariate_normal_cdf(0, 0, 0.5), 1.0 / 3, 1e-15);
    EXPECT_NEAR(bivariate_normal_cdf(0.4, -0.2, 0), 0.5 * std::erfc(-0.4 / std::sqrt(2)) * 0.5 *
                                                        std::erfc(0.2 / std::sqrt(2)), 1e-15);
    EXPECT_NEAR(bivariate_normal_cdf(INFINITY, 0.3, 0.7), 0.5 * std::erfc(-0.3 / std::sqrt(2)), 1e-15);
    EXPECT_EQ(bivariate_normal_cdf(-INFINITY, 0.3, 0.7), 0.0);
    EXPECT_NEAR(bivariate_normal_cdf(0.5, 1.0, 1.0), 0.5 * std::erfc(-0.5 / std::sqrt(2)), 1e-15);
    EXPECT_THROW(bivariate_normal_cdf(0, 0, 1.5), std::invalid_argument);
}

TEST(Polychoric, TwoStepMatchesReferences) {
    const auto p = polychoric_correlation(ints(pc_x), ints(pc_y));
    ASSERT_TRUE(p.converged);
    EXPECT_NEAR(p.rho, pc_two[0], 2e-8); // R's optimize with pmvnorm cells: ~1e-8
    EXPECT_NEAR(p.standard_error / pc_two[1], 1.0, 1e-6);
    EXPECT_NEAR(p.chi_square, pc_two[2], 1e-8);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(p.row_thresholds[i], pc_two[3 + i], 1e-14);
    for (std::size_t j = 0; j < 2; ++j)
        EXPECT_NEAR(p.column_thresholds[j], pc_two[6 + j], 1e-14);
    EXPECT_EQ(p.df, pc_polycor[3]);
    EXPECT_NEAR(p.rho, pc_polycor[0], 1e-4); // polycor's optim is less precise
    EXPECT_NEAR(p.chi_square, pc_polycor[2], 1e-4);
    // Same table directly.
    EXPECT_NEAR(polychoric_correlation(table(pc_table, 3)).rho, p.rho, 1e-14);
    const auto tet = tetrachoric_correlation(40, 10, 10, 40);
    EXPECT_NEAR(tet.rho, pc_tet[0], 2e-8);
    // Symmetric margins: P(both below 0) = 0.4 = 1/4 + asin(rho)/(2 pi), so rho = sin(0.3 pi) exactly.
    EXPECT_NEAR(tet.rho, std::sin(0.3 * std::numbers::pi), 1e-13);
    EXPECT_NEAR(tet.standard_error / pc_tet[1], 1.0, 1e-6);
    EXPECT_TRUE(std::isnan(tet.p_value));
}

TEST(Polychoric, MaximumLikelihoodMatchesReferences) {
    for (const auto& [tab, cols, ref] : {std::tuple{&pc_table, 3uz, &pc_ml}, std::tuple{&pc_zero_table, 3uz, &pc_zero_ml}}) {
        const auto t = table(*tab, cols);
        const auto p = polychoric_correlation(t, {.estimator = PolychoricEstimator::maximum_likelihood});
        ASSERT_TRUE(p.converged);
        const std::size_t r = t.size(), q = r + cols - 1;
        EXPECT_NEAR(p.rho, (*ref)[0], 2e-6);
        for (std::size_t i = 0; i + 1 < r; ++i) {
            EXPECT_NEAR(p.row_thresholds[i], (*ref)[1 + i], 2e-6);
            EXPECT_NEAR(p.row_threshold_se[i] / (*ref)[q + 1 + i], 1.0, 1e-4);
        }
        for (std::size_t j = 0; j + 1 < cols; ++j) {
            EXPECT_NEAR(p.column_thresholds[j], (*ref)[r + j], 2e-6);
            EXPECT_NEAR(p.column_threshold_se[j] / (*ref)[q + r + j], 1.0, 1e-4);
        }
        EXPECT_NEAR(p.standard_error / (*ref)[q], 1.0, 1e-4);
        EXPECT_LE(p.chi_square, (*ref)[2 * q] + 1e-9); // at least as good an optimum
        EXPECT_NEAR(p.chi_square, (*ref)[2 * q], 1e-6);
    }
    const auto two = polychoric_correlation(table(pc_zero_table, 3));
    EXPECT_NEAR(two.rho, pc_zero_two[0], 2e-8);
    EXPECT_NEAR(two.standard_error / pc_zero_two[1], 1.0, 1e-6);
}

TEST(Polychoric, MatrixAndErrors) {
    const auto m = polychoric_matrix({ints(pc_x), ints(pc_y), ints(pc_x)});
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_NEAR(m(0, 1), pc_two[0], 2e-8);
    EXPECT_NEAR(m(1, 0), m(0, 1), 0.0);
    EXPECT_GT(m(0, 2), 0.999); // identical variables hit the admissibility bound
    EXPECT_THROW(polychoric_correlation({{1, 2}, {3}}), std::invalid_argument);
    EXPECT_THROW(polychoric_correlation({{1, 2}, {0, 0}}), std::invalid_argument);
    EXPECT_THROW(polychoric_correlation({{1, -2}, {3, 4}}), std::invalid_argument);
    EXPECT_THROW(polychoric_correlation(std::vector<int>{1, 1, 1}, std::vector<int>{1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(polychoric_matrix({ints(pc_x)}), std::invalid_argument);
}
