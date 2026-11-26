#include <datamunge/stats/copula.hpp>
#include <datamunge/stats/correlation_test.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_copula_data.inc"

Matrix rows(const std::vector<double>& v, std::size_t cols) {
    Matrix m(v.size() / cols, cols, 0.0);
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < cols; ++j)
            m(i, j) = v[i * cols + j];
    return m;
}

Matrix corr3(double r12, double r13, double r23) { return Matrix(3, 3, {1, r12, r13, r12, 1, r23, r13, r23, 1}); }

void expect_pairs(const Matrix& r, const std::vector<double>& ref, double tol) {
    EXPECT_NEAR(r(0, 1), ref[0], tol);
    EXPECT_NEAR(r(0, 2), ref[1], tol);
    EXPECT_NEAR(r(1, 2), ref[2], tol);
}
} // namespace

TEST(Copula, PseudoObservationsAndDensities) {
    const auto u = pseudo_observations(rows(cp_x, 3));
    const auto ref = rows(cp_u, 3);
    for (std::size_t i = 0; i < u.rows(); ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(u(i, j), ref(i, j), 1e-15);
    const auto r = corr3(0.5, 0.2, -0.3);
    for (std::size_t i = 0; i < 3; ++i) {
        const std::vector<double> p{cp_points[3 * i], cp_points[3 * i + 1], cp_points[3 * i + 2]};
        EXPECT_NEAR(copula_log_density(p, r, CopulaFamily::student_t, 3.5), cp_t_density[i], 1e-11);
        EXPECT_NEAR(copula_log_density(p, r, CopulaFamily::gaussian), cp_gauss_density[i], 1e-12);
    }
    EXPECT_NEAR(t_copula_tail_dependence(0.5, 4), cp_lambda[0], 1e-13);
    EXPECT_NEAR(t_copula_tail_dependence(-0.3, 2.5), cp_lambda[1], 1e-13);
    EXPECT_NEAR(t_copula_tail_dependence(0.9, 10), cp_lambda[2], 1e-13);
    EXPECT_EQ(t_copula_tail_dependence(0.9, INFINITY), 0.0);
}

TEST(Copula, FitsMatchCopulaPackage) {
    const auto u = rows(cp_u, 3);
    const auto gi = fit_copula(u, {.method = CopulaMethod::kendall_inversion});
    expect_pairs(gi.correlation, cp_gauss_itau, 1e-14);
    EXPECT_FALSE(gi.projected);
    const auto g = fit_copula(u);
    expect_pairs(g.correlation, cp_gauss_mpl, 2e-6);
    EXPECT_GE(g.log_likelihood, cp_gauss_mpl[3] - 1e-9);
    EXPECT_NEAR(g.log_likelihood, cp_gauss_mpl[3], 1e-7);
    EXPECT_TRUE(std::isinf(g.df));
    const auto t = fit_copula(u, {.family = CopulaFamily::student_t});
    expect_pairs(t.correlation, cp_t_mpl, 2e-5);
    EXPECT_NEAR(t.df, cp_t_mpl[3], 1e-3);
    EXPECT_GE(t.log_likelihood, cp_t_mpl[4] - 1e-8);
    EXPECT_NEAR(t.log_likelihood, cp_t_mpl[4], 1e-6);
    EXPECT_EQ(t.parameters, 4u);
    EXPECT_NEAR(t.aic, -2 * t.log_likelihood + 8, 1e-12);
    EXPECT_NEAR(t.tail_dependence(0, 1), t_copula_tail_dependence(t.correlation(0, 1), t.df), 1e-15);
    const auto ti = fit_copula(u, {.family = CopulaFamily::student_t, .method = CopulaMethod::kendall_inversion});
    expect_pairs(ti.correlation, cp_t_itau, 1e-14);
    EXPECT_NEAR(ti.df, cp_t_itau[3], 1e-3);
    EXPECT_NEAR(ti.log_likelihood, cp_t_itau[4], 1e-6);
    const auto fixed = fit_copula(u, {.family = CopulaFamily::student_t, .df = 4});
    EXPECT_EQ(fixed.df, 4.0);
    EXPECT_EQ(fixed.parameters, 3u);
}

TEST(Copula, SimulationAndErrors) {
    const auto r = corr3(0.7, 0.3, 0.1);
    const auto u = simulate_copula(r, CopulaFamily::student_t, 3, 8000, 5);
    std::vector<double> a(u.rows()), b(u.rows());
    for (std::size_t i = 0; i < u.rows(); ++i)
        a[i] = u(i, 0), b[i] = u(i, 1);
    // Kendall's tau of any elliptical copula is (2/pi) asin(rho).
    EXPECT_NEAR(kendall_tau(a, b).tau_b, 2 / std::numbers::pi * std::asin(0.7), 0.015);
    const auto fit = fit_copula(pseudo_observations(u), {.family = CopulaFamily::student_t});
    EXPECT_NEAR(fit.df, 3.0, 0.6);
    EXPECT_NEAR(fit.correlation(0, 1), 0.7, 0.03);
    EXPECT_THROW(fit_copula(Matrix(5, 1, 0.5)), std::invalid_argument);
    EXPECT_THROW(fit_copula(Matrix(5, 2, 1.0)), std::invalid_argument);
    EXPECT_THROW(copula_log_density({0.5, 0.5}, Matrix(2, 2, {1, 1, 1, 1}), CopulaFamily::gaussian),
                 std::invalid_argument);
    EXPECT_THROW(simulate_copula(r, CopulaFamily::student_t, 0, 5, 1), std::invalid_argument);
    EXPECT_THROW(t_copula_tail_dependence(-1, 3), std::invalid_argument);
}
