#include <datamunge/stats/gmm_estimation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using datamunge::stats::generalized_method_of_moments;
using datamunge::stats::GmmEstimator;
using datamunge::stats::GmmOptions;
using datamunge::stats::GmmResult;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_gmm_estimation_data.inc"

// Layout: coefficients, SEs, J, J p-value.
void expect_gmm(const GmmResult& r, const std::vector<double>& ref, double tol, const std::string& what) {
    const std::size_t k = r.coefficients.size();
    for (std::size_t a = 0; a < k; ++a) {
        EXPECT_NEAR(r.coefficients[a], ref[a], tol * (1 + std::abs(ref[a]))) << what << " coef " << a;
        EXPECT_NEAR(r.standard_errors[a], ref[k + a], tol * ref[k + a]) << what << " se " << a;
    }
    EXPECT_NEAR(r.j_statistic, ref[2 * k], tol * (1 + ref[2 * k])) << what << " J";
    EXPECT_NEAR(r.j_p_value, ref[2 * k + 1], tol) << what << " J p";
}

Matrix iv_moments(const std::vector<double>& t) {
    const std::size_t n = iv_y.size();
    Matrix g(n, 4);
    for (std::size_t i = 0; i < n; ++i) {
        const double e = iv_y[i] - t[0] - t[1] * iv_x[i];
        g(i, 0) = e;
        for (std::size_t j = 0; j < 3; ++j)
            g(i, j + 1) = iv_z[3 * i + j] * e;
    }
    return g;
}

Matrix gamma_moments(const std::vector<double>& t) {
    const double a = t[0], b = t[1];
    Matrix g(gamma_x.size(), 3);
    for (std::size_t i = 0; i < gamma_x.size(); ++i) {
        const double x = gamma_x[i];
        g(i, 0) = x - a / b;
        g(i, 1) = x * x - a * (a + 1) / (b * b);
        g(i, 2) = x * x * x - a * (a + 1) * (a + 2) / (b * b * b);
    }
    return g;
}

Matrix gamma_jacobian(const std::vector<double>& t) {
    const double a = t[0], b = t[1];
    return Matrix(3, 2, {-1 / b, a / (b * b), -(2 * a + 1) / (b * b), 2 * a * (a + 1) / (b * b * b),
                         -(3 * a * a + 6 * a + 2) / (b * b * b), 3 * a * (a + 1) * (a + 2) / (b * b * b * b)});
}
} // namespace

TEST(Gmm, LinearIvMatchesGmmPackage) {
    // For linear models gmm's first step is 2SLS: initial weights (Z'Z / n)^{-1}.
    Matrix zz(4, 4, 0.0);
    const std::size_t n = iv_y.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double zi[4]{1, iv_z[3 * i], iv_z[3 * i + 1], iv_z[3 * i + 2]};
        for (std::size_t a = 0; a < 4; ++a)
            for (std::size_t b = 0; b < 4; ++b)
                zz(a, b) += zi[a] * zi[b] / static_cast<double>(n);
    }
    Matrix inv(4, 4, 0.0);
    for (std::size_t c = 0; c < 4; ++c) { // Gauss-Jordan on a copy
        std::vector<double> e(4, 0.0);
        e[c] = 1;
        Matrix a = zz;
        for (std::size_t p = 0; p < 4; ++p)
            for (std::size_t r = 0; r < 4; ++r)
                if (r != p) {
                    const double f = a(r, p) / a(p, p);
                    for (std::size_t j = 0; j < 4; ++j)
                        a(r, j) -= f * a(p, j);
                    e[r] -= f * e[p];
                }
        for (std::size_t r = 0; r < 4; ++r)
            inv(r, c) = e[r] / a(r, r);
    }
    expect_gmm(generalized_method_of_moments(iv_moments, {0, 0}, {.initial_weights = inv}), ref_two_step, 1e-8,
               "two-step");
    // The default identity first step gives a different, equally valid two-step estimate.
    const auto identity_first = generalized_method_of_moments(iv_moments, {0, 0});
    EXPECT_NEAR(identity_first.coefficients[1], ref_two_step[1], 1e-3);
    expect_gmm(generalized_method_of_moments(iv_moments, {0, 0}, {.estimator = GmmEstimator::iterated}),
               ref_iterated, 1e-8, "iterated");
    expect_gmm(generalized_method_of_moments(iv_moments, {0, 0}, {.estimator = GmmEstimator::one_step}),
               ref_identity, 1e-8, "identity");
}

TEST(Gmm, NonlinearMomentsMatchGmmPackage) {
    const auto analytic =
        generalized_method_of_moments(gamma_moments, {2, 1.5}, {.jacobian = gamma_jacobian});
    expect_gmm(analytic, ref_gamma, 1e-6, "gamma analytic");
    EXPECT_TRUE(analytic.converged);
    const auto numeric = generalized_method_of_moments(gamma_moments, {2, 1.5});
    expect_gmm(numeric, ref_gamma, 1e-6, "gamma numeric");
}

TEST(Gmm, ExactlyIdentifiedAndFixedWeights) {
    // Just-identified IV (one instrument): J is zero and the weights do not matter.
    const auto g2 = [](const std::vector<double>& t) {
        const Matrix full = iv_moments(t);
        Matrix g(full.rows(), 2);
        for (std::size_t i = 0; i < full.rows(); ++i) {
            g(i, 0) = full(i, 0);
            g(i, 1) = full(i, 1);
        }
        return g;
    };
    const auto a = generalized_method_of_moments(g2, {0, 0});
    const auto b = generalized_method_of_moments(g2, {0, 0}, {.estimator = GmmEstimator::one_step});
    EXPECT_EQ(a.df, 0u);
    EXPECT_EQ(a.j_statistic, 0.0);
    EXPECT_NEAR(a.coefficients[1], b.coefficients[1], 1e-10);
    EXPECT_NEAR(a.standard_errors[1], b.standard_errors[1], 1e-10);
    for (const double m : a.mean_moments)
        EXPECT_NEAR(m, 0.0, 1e-10);
    // Fixed first-step weights equal to the identity reproduce the identity fit.
    Matrix eye(4, 4, 0.0);
    for (std::size_t i = 0; i < 4; ++i)
        eye(i, i) = 1;
    const auto fixed = generalized_method_of_moments(
        iv_moments, {0, 0}, {.estimator = GmmEstimator::one_step, .initial_weights = eye});
    expect_gmm(fixed, ref_identity, 1e-8, "fixed identity");
}

TEST(Gmm, ValidatesInputs) {
    EXPECT_THROW(generalized_method_of_moments(iv_moments, {}), std::invalid_argument);
    EXPECT_THROW(generalized_method_of_moments(iv_moments, {0, 0, 0, 0, 0}), std::invalid_argument);
    EXPECT_THROW(generalized_method_of_moments(iv_moments, {0, 0}, {.initial_weights = Matrix(2, 2, 1.0)}),
                 std::invalid_argument);
    EXPECT_THROW(generalized_method_of_moments(iv_moments, {0, 0}, {.confidence = 1.5}), std::invalid_argument);
    const auto nan_moments = [](const std::vector<double>&) { return Matrix(10, 2, std::nan("")); };
    EXPECT_THROW(generalized_method_of_moments(nan_moments, {0}), std::runtime_error);
}
