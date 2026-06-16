#include <gtest/gtest.h>

#include <datamunge/linalg/linalg.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::linalg;

TEST(LinearRegression, DenseRecoversNoiselessParameters) {
    DenseMatrix<double> X(6, 3, {
        1, 0,  1,
        1, 1, -1,
        1, 2,  0,
        1, 3,  2,
        1, 4, -2,
        1, 5,  1,
    });
    const std::vector<double> beta = {1.5, -0.25, 2.0};

    std::vector<double> y(X.rows(), 0.0);
    for (std::size_t i = 0; i < X.rows(); ++i)
        y[i] = beta[0] + beta[1] * X(i, 1) + beta[2] * X(i, 2);

    const auto fit = linear_regression(X, y);

    ASSERT_EQ(fit.coefficients.size(), beta.size());
    for (std::size_t j = 0; j < beta.size(); ++j)
        EXPECT_NEAR(fit.coefficients[j], beta[j], 1e-10);

    EXPECT_NEAR(fit.residual_sum_of_squares, 0.0, 1e-12);
    EXPECT_NEAR(fit.sigma2, 0.0, 1e-12);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-12);
    EXPECT_EQ(fit.degrees_of_freedom, 3u);
}

TEST(LinearRegression, SparseMatchesDenseOnSparseDesign) {
    DenseMatrix<double> X_dense(8, 4, 0.0);
    SparseCOO<double>   X_sparse(8, 4, 20);
    const std::vector<double> beta = {0.75, -1.25, 0.5, 2.0};
    const std::vector<double> noise = {0.10, -0.20, 0.05, -0.15, 0.00, 0.12, -0.08, 0.18};

    for (std::size_t i = 0; i < 8; ++i) {
        const double x1 = (i == 0 || i == 4) ? 0.0 : static_cast<double>(i) - 2.0;
        const double x2 = (i % 2 == 0) ? 0.0 : 1.0;
        const double x3 = (i % 3 == 0) ? 0.0 : -1.0;
        const double row[4] = {1.0, x1, x2, x3};
        for (std::size_t j = 0; j < 4; ++j) {
            X_dense(i, j) = row[j];
            if (row[j] != 0.0) X_sparse.set(i, j, row[j]);
        }
    }
    X_sparse.compress();

    std::vector<double> y(8, 0.0);
    for (std::size_t i = 0; i < 8; ++i) {
        y[i] = beta[0] + beta[1] * X_dense(i, 1) + beta[2] * X_dense(i, 2)
             + beta[3] * X_dense(i, 3) + noise[i];
    }

    const auto dense_fit  = linear_regression(X_dense, y);
    SparseLinearRegressionOptions opts;
    opts.solver.tol      = 1e-12;
    opts.solver.max_iter = 64;
    const auto sparse_fit = sparse_linear_regression(X_sparse, y, opts);

    ASSERT_EQ(dense_fit.coefficients.size(), sparse_fit.coefficients.size());
    for (std::size_t j = 0; j < dense_fit.coefficients.size(); ++j) {
        EXPECT_NEAR(sparse_fit.coefficients[j], dense_fit.coefficients[j], 1e-8);
        EXPECT_NEAR(sparse_fit.standard_errors[j], dense_fit.standard_errors[j], 1e-8);
        EXPECT_NEAR(sparse_fit.parameter_variances[j], dense_fit.parameter_variances[j], 1e-8);
    }

    EXPECT_NEAR(sparse_fit.sigma2, dense_fit.sigma2, 1e-10);
    EXPECT_NEAR(sparse_fit.r_squared, dense_fit.r_squared, 1e-10);
}

TEST(LinearRegression, RejectsZeroDegreesOfFreedom) {
    DenseMatrix<double> X(3, 3, {
        1, 0, 0,
        1, 1, 0,
        1, 0, 1,
    });
    const std::vector<double> y = {1.0, 2.0, 3.0};

    EXPECT_THROW(linear_regression(X, y), std::invalid_argument);

    SparseCOO<double> X_sparse(3, 3, 5);
    X_sparse.set(0, 0, 1.0);
    X_sparse.set(1, 0, 1.0);
    X_sparse.set(1, 1, 1.0);
    X_sparse.set(2, 0, 1.0);
    X_sparse.set(2, 2, 1.0);
    X_sparse.compress();

    EXPECT_THROW(sparse_linear_regression(X_sparse, y), std::invalid_argument);
}
