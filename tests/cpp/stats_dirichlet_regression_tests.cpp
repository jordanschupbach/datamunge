#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/beta_regression.hpp>
#include <datamunge/stats/dirichlet_regression.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_dirichlet_regression_data.inc"

struct Arctic {
    Matrix y{39, 3, 0.0}, depth_design1{39, 3, 0.0}, depth_design{39, 2, 0.0}, one{39, 1, 1.0};
    Arctic() {
        for (std::size_t i = 0; i < 39; ++i) {
            const double d = dr_arctic[4 * i + 3];
            for (std::size_t j = 0; j < 3; ++j)
                y(i, j) = dr_arctic[4 * i + j];
            depth_design1(i, 0) = 1, depth_design1(i, 1) = d, depth_design1(i, 2) = d * d;
            depth_design(i, 0) = 1, depth_design(i, 1) = d;
        }
    }
};

std::vector<double> flatten(const DirichletRegression& f) {
    std::vector<double> v;
    for (std::size_t j = 0; j < f.mean_coefficients.cols(); ++j)
        if (j != f.base)
            for (std::size_t c = 0; c < f.mean_coefficients.rows(); ++c)
                v.push_back(f.mean_coefficients(c, j));
    v.insert(v.end(), f.precision_coefficients.begin(), f.precision_coefficients.end());
    return v;
}

std::vector<double> flatten_se(const DirichletRegression& f) {
    std::vector<double> v;
    for (std::size_t j = 0; j < f.mean_standard_errors.cols(); ++j)
        if (j != f.base)
            for (std::size_t c = 0; c < f.mean_standard_errors.rows(); ++c)
                v.push_back(f.mean_standard_errors(c, j));
    v.insert(v.end(), f.precision_standard_errors.begin(), f.precision_standard_errors.end());
    return v;
}
} // namespace

TEST(DirichletRegression, SpecialFunctions) {
    EXPECT_NEAR(datamunge::random::digamma(1.0), -std::numbers::egamma, 1e-15);
    EXPECT_NEAR(datamunge::random::trigamma(1.0), std::numbers::pi * std::numbers::pi / 6, 1e-15);
    EXPECT_NEAR(datamunge::random::digamma(0.5), -std::numbers::egamma - 2 * std::numbers::ln2, 1e-15);
    EXPECT_NEAR(datamunge::random::trigamma(0.5), std::numbers::pi * std::numbers::pi / 2, 1e-14);
    EXPECT_TRUE(std::isnan(datamunge::random::digamma(-2.0)));
}

TEST(DirichletRegression, MatchesDirichletReg) {
    const Arctic a;
    const auto m1 = dirichlet_regression(a.depth_design1, a.depth_design, a.y, {.base = 2});
    ASSERT_TRUE(m1.converged);
    const auto c1 = flatten(m1), s1 = flatten_se(m1);
    for (std::size_t r = 0; r < c1.size(); ++r) {
        EXPECT_NEAR(c1[r] / dr_m1_coef[r], 1.0, 1e-7) << r;
        EXPECT_NEAR(s1[r] / dr_m1_se[r], 1.0, 1e-6) << r;
    }
    EXPECT_NEAR(m1.log_likelihood, dr_m1_ll[0], 1e-9);
    EXPECT_NEAR(m1.aic, dr_m1_ll[1], 1e-8);
    EXPECT_NEAR(m1.bic, dr_m1_ll[2], 1e-8);
    EXPECT_FALSE(m1.compressed);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(m1.fitted.precision[i], dr_m1_phi[i], 1e-6);
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(m1.fitted.mean(i, j), dr_m1_fitted[3 * i + j], 1e-8);
            EXPECT_NEAR(m1.standardized_residuals(i, j), dr_m1_resid[3 * i + j], 1e-6);
        }
    }
    EXPECT_EQ(m1.mean_coefficients(0, 2), 0.0);
    EXPECT_TRUE(std::isnan(m1.mean_standard_errors(0, 2)));
    const auto pred = m1.predict(a.depth_design1, a.depth_design);
    EXPECT_NEAR(pred.mean(7, 1), m1.fitted.mean(7, 1), 1e-15);

    const auto m2 = dirichlet_regression(a.depth_design, a.one, a.y, {.base = 0});
    const auto c2 = flatten(m2), s2 = flatten_se(m2);
    for (std::size_t r = 0; r < c2.size(); ++r) {
        EXPECT_NEAR(c2[r] / dr_m2_coef[r], 1.0, 1e-7) << r;
        EXPECT_NEAR(s2[r] / dr_m2_se[r], 1.0, 1e-6) << r;
    }
    EXPECT_NEAR(m2.log_likelihood, dr_m2_ll[0], 1e-9);
}

TEST(DirichletRegression, BoundaryCompressionAndBetaCase) {
    Arctic a;
    a.y(4, 2) = 0;
    const auto m3 = dirichlet_regression(a.depth_design, a.depth_design, a.y);
    EXPECT_TRUE(m3.compressed);
    const auto c3 = flatten(m3);
    for (std::size_t r = 0; r < c3.size(); ++r)
        EXPECT_NEAR(c3[r] / dr_m3_coef[r], 1.0, 1e-7) << r;
    EXPECT_NEAR(m3.log_likelihood, dr_m3_ll[0], 1e-9);
    EXPECT_THROW(dirichlet_regression(a.depth_design, a.depth_design, a.y, {.compress_boundaries = false}),
                 std::invalid_argument);
    EXPECT_THROW(dirichlet_regression(a.depth_design, a.depth_design, a.y, {.base = 3}), std::invalid_argument);

    // D = 2 with the second part as base is beta regression with a logit mean.
    const Arctic b;
    Matrix two(39, 2, 0.0);
    std::vector<double> sand(39);
    for (std::size_t i = 0; i < 39; ++i) {
        const double s = b.y(i, 0) / (b.y(i, 0) + b.y(i, 1) + b.y(i, 2));
        two(i, 0) = s, two(i, 1) = 1 - s, sand[i] = s;
    }
    const auto dir = dirichlet_regression(b.depth_design, b.depth_design, two, {.base = 1});
    const auto beta = beta_regression(b.depth_design, b.depth_design, sand);
    for (std::size_t c = 0; c < 2; ++c) {
        EXPECT_NEAR(dir.mean_coefficients(c, 0), beta.mean_coefficients[c], 1e-7);
        EXPECT_NEAR(dir.precision_coefficients[c], beta.precision_coefficients[c], 1e-7);
    }
    EXPECT_NEAR(dir.log_likelihood, beta.log_likelihood, 1e-8);
}
