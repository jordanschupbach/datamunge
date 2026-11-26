#include <datamunge/stats/reliability.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_reliability_data.inc"

Matrix matrix(const std::vector<double>& x, std::size_t n, std::size_t p) {
    Matrix m(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            m(i, j) = x[i * p + j];
    return m;
}

void check_alpha(const Matrix& x, const std::vector<double>& ref, const std::vector<double>& rdrop,
                 const std::vector<double>* adrop) {
    const auto a = cronbach_alpha(x);
    EXPECT_NEAR(a.alpha, ref[0], 1e-12);
    EXPECT_NEAR(a.standardized_alpha, ref[1], 1e-12);
    EXPECT_NEAR(a.average_r, ref[2], 1e-12);
    EXPECT_NEAR(a.standard_error, ref[3], 1e-12);
    EXPECT_NEAR(a.feldt_lower, ref[4], 1e-9);
    EXPECT_NEAR(a.feldt_upper, ref[5], 1e-9);
    for (std::size_t j = 0; j < x.cols(); ++j) {
        EXPECT_NEAR(a.item_statistics[j].item_rest, rdrop[j], 1e-12) << j;
        if (adrop)
            EXPECT_NEAR(a.item_statistics[j].alpha_if_deleted, (*adrop)[j], 1e-12) << j;
    }
}

void check_omega(const Matrix& x, const std::vector<double>& lambda, const std::vector<double>& psi,
                 const std::vector<double>& ref) {
    const auto o = mcdonald_omega(x, {.replicates = 0});
    ASSERT_TRUE(o.fit.converged);
    for (std::size_t j = 0; j < x.cols(); ++j) {
        EXPECT_NEAR(o.fit.loadings[j], lambda[j], 2e-5) << j; // lavaan optimiser tolerance
        EXPECT_NEAR(o.fit.uniquenesses[j], psi[j], 2e-5) << j;
    }
    EXPECT_NEAR(o.omega, ref[0], 1e-6);
    EXPECT_NEAR(o.omega_observed, ref[1], 1e-6);
    EXPECT_NEAR(o.fit.chi_square, ref[2], 1e-5);
    EXPECT_EQ(o.fit.df, ref[3]);
    EXPECT_NEAR(o.fit.p_value, ref[4], 1e-6);
    EXPECT_TRUE(std::isnan(o.omega_lower));
}
} // namespace

TEST(CronbachAlpha, MatchesPsych) {
    check_alpha(matrix(c6_x, c6_n, c6_p), c6_alpha, c6_rdrop, &c6_adrop);
    check_alpha(matrix(o4_x, o4_n, o4_p), o4_alpha, o4_rdrop, &o4_adrop);
    check_alpha(matrix(t2_x, t2_n, t2_p), t2_alpha, t2_rdrop, nullptr);
}

TEST(McDonaldOmega, MatchesLavaan) {
    check_omega(matrix(c6_x, c6_n, c6_p), c6_lambda, c6_psi, c6_omega);
    check_omega(matrix(o4_x, o4_n, o4_p), o4_lambda, o4_psi, o4_omega);
}

TEST(McDonaldOmega, TauEquivalentAndBootstrap) {
    // Exactly compound-symmetric covariance: one factor fits perfectly and omega equals alpha.
    const Matrix s(4, 4, {2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2});
    const auto f = one_factor_fit(s, 101);
    for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(f.loadings[j], std::sqrt(100.0 / 101), 1e-9);
        EXPECT_NEAR(f.uniquenesses[j], 100.0 / 101, 1e-9);
    }
    EXPECT_NEAR(f.chi_square, 0.0, 1e-9);
    const Matrix x = matrix(c6_x, c6_n, c6_p);
    const auto o = mcdonald_omega(x, {.replicates = 300, .seed = 4});
    EXPECT_EQ(o.replicates, 300u);
    EXPECT_LT(o.omega_lower, o.omega);
    EXPECT_GT(o.omega_upper, o.omega);
    EXPECT_LT(o.alpha_lower, o.alpha);
    EXPECT_GT(o.alpha_upper, o.alpha);
    EXPECT_GT(o.omega, o.alpha); // congeneric items: alpha is a lower bound
    EXPECT_THROW(cronbach_alpha(Matrix(5, 1, 1.0)), std::invalid_argument);
    EXPECT_THROW(cronbach_alpha(Matrix(5, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW(mcdonald_omega(matrix(t2_x, t2_n, t2_p)), std::invalid_argument);
}
