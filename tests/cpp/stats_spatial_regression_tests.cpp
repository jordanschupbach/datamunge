#include <datamunge/stats/spatial_regression.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_spatial_regression_data.inc"

const std::size_t n = 49;
Matrix design() {
    Matrix x(n, 3);
    for (std::size_t i = 0; i < n; ++i)
        x(i, 0) = 1, x(i, 1) = inc[i], x(i, 2) = hoval[i];
    return x;
}
const Matrix binary(n, n, col_binary);

void expect_fit(const SpatialRegressionResult& f, const std::vector<double>& r, const std::string& what) {
    EXPECT_NEAR(f.spatial_parameter, r[0], 1e-6) << what;
    EXPECT_NEAR(f.spatial_se, r[1], 1e-6) << what;
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(f.coefficients[k], r[2 + k], 1e-5 * (1 + std::abs(r[2 + k]))) << what << " beta " << k;
        EXPECT_NEAR(f.standard_errors[k], r[5 + k], 1e-5 * r[5 + k]) << what << " se " << k;
    }
    EXPECT_NEAR(f.sigma2, r[8], 1e-5 * r[8]) << what;
    EXPECT_NEAR(f.log_likelihood, r[9], 1e-8) << what;
    EXPECT_NEAR(f.ols_log_likelihood, r[10], 1e-9) << what;
    EXPECT_NEAR(f.aic, r[11], 1e-7) << what;
}
} // namespace

TEST(SpatialRegression, LagMatchesSpatialreg) {
    for (const bool standardise : {true, false}) {
        const Matrix w = standardise ? row_standardize(binary) : binary;
        const auto& ref = standardise ? lag_W : lag_B;
        const auto& fit_ref = standardise ? lag_W_fitted : lag_B_fitted;
        const auto& imp = standardise ? lag_W_impacts : lag_B_impacts;
        const auto f = spatial_lag_regression(crime, design(), w);
        expect_fit(f, ref, standardise ? "lag W" : "lag B");
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(f.fitted[i], fit_ref[i], 1e-5) << i;
        for (std::size_t k = 0; k < 2; ++k) {
            EXPECT_NEAR(f.impacts[k + 1].direct, imp[k], 1e-5) << k;
            // spatialreg's total is beta / (1 - rho), exact only for row-standardised weights.
            const double total = standardise ? imp[4 + k] : lag_B_total_rowsum[k];
            EXPECT_NEAR(f.impacts[k + 1].total, total, 1e-5) << k;
            EXPECT_NEAR(f.impacts[k + 1].indirect, total - imp[k], 1e-5) << k;
        }
        EXPECT_NEAR(f.lr_statistic, 2 * (ref[9] - ref[10]), 1e-7);
    }
}

TEST(SpatialRegression, ErrorMatchesSpatialreg) {
    for (const bool standardise : {true, false}) {
        const Matrix w = standardise ? row_standardize(binary) : binary;
        const auto f = spatial_error_regression(crime, design(), w);
        expect_fit(f, standardise ? err_W : err_B, standardise ? "error W" : "error B");
        const auto& fit_ref = standardise ? err_W_fitted : err_B_fitted;
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(f.fitted[i], fit_ref[i], 1e-5) << i;
        EXPECT_TRUE(f.impacts.empty());
    }
}

TEST(SpatialRegression, AsymmetricWeightsUseLu) {
    const Matrix xy(n, 2, col_xy);
    const auto w = row_standardize(knn_weights(xy, 4));
    const auto l = spatial_lag_regression(crime, design(), w);
    EXPECT_NEAR(l.spatial_parameter, lag_knn[0], 1e-6);
    EXPECT_NEAR(l.log_likelihood, lag_knn[4], 1e-8);
    for (std::size_t k = 0; k < 3; ++k)
        EXPECT_NEAR(l.coefficients[k], lag_knn[1 + k], 1e-5 * (1 + std::abs(lag_knn[1 + k])));
    const auto e = spatial_error_regression(crime, design(), w);
    EXPECT_NEAR(e.spatial_parameter, err_knn[0], 1e-6);
    EXPECT_NEAR(e.log_likelihood, err_knn[4], 1e-8);
    EXPECT_DOUBLE_EQ(l.lower, -1);
    EXPECT_DOUBLE_EQ(l.upper, 1);
}

TEST(SpatialRegression, Validates) {
    const auto w = row_standardize(binary);
    EXPECT_THROW(spatial_lag_regression(std::vector<double>(3, 1.0), design(), w), std::invalid_argument);
    Matrix bad = w;
    bad(0, 0) = 0.1;
    EXPECT_THROW(spatial_error_regression(crime, design(), bad), std::invalid_argument);
    EXPECT_THROW(spatial_lag_regression(crime, design(), w, {.lower = 0.5, .upper = 0.2}), std::invalid_argument);
    Matrix collinear = design();
    for (std::size_t i = 0; i < n; ++i)
        collinear(i, 2) = 2 * collinear(i, 1);
    EXPECT_THROW(spatial_lag_regression(crime, collinear, w), std::runtime_error);
}
