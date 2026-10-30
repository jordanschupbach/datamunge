#include <gtest/gtest.h>

#include <datamunge/linalg/dense.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::linalg;

namespace {

DenseMatrix<double> random_matrix(std::size_t m, std::size_t n, unsigned seed) {
    std::mt19937_64                  rng(seed);
    std::normal_distribution<double> z;
    DenseMatrix<double>              A(m, n);
    for (auto& v : A.data()) v = z(rng);
    return A;
}

double norm_of_column(const DenseMatrix<double>& A, std::size_t j) {
    double s = 0;
    for (std::size_t i = 0; i < A.rows(); ++i) s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

double max_abs_diff(const DenseMatrix<double>& A, const DenseMatrix<double>& B) {
    return (A - B).norm_max();
}

// Column j of A scaled by 10^(-j * step): kappa grows geometrically.
DenseMatrix<double> graded(std::size_t m, std::size_t n, double step, unsigned seed) {
    auto A = random_matrix(m, n, seed);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) *= std::pow(10.0, -step * static_cast<double>(j));
    return A;
}

// U diag(sigma) V^T with random orthogonal U (m x n) and V: near-collinear columns.
DenseMatrix<double> with_singular_values(std::size_t m, const std::vector<double>& sigma, unsigned seed) {
    const std::size_t n = sigma.size();
    const auto U = qr(random_matrix(m, n, seed)).Q;
    const auto V = qr(random_matrix(n, n, seed + 1)).Q;
    DenseMatrix<double> A(m, n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k) A(i, j) += U(i, k) * sigma[k] * V(j, k);
    return A;
}

} // namespace

TEST(Svd, ReconstructsAndIsOrthonormal) {
    for (const auto [m, n] : {std::pair{7u, 4u}, std::pair{4u, 7u}, std::pair{5u, 5u}, std::pair{1u, 3u}}) {
        const auto A = random_matrix(m, n, 11 + m * 10 + n);
        const auto f = svd(A);
        const std::size_t k = std::min(m, n);
        ASSERT_EQ(f.s.size(), k);
        EXPECT_EQ(f.U.rows(), m);
        EXPECT_EQ(f.V.rows(), n);
        EXPECT_TRUE(f.converged);
        for (std::size_t j = 1; j < k; ++j) EXPECT_GE(f.s[j - 1], f.s[j]);
        EXPECT_LT(max_abs_diff(f.reconstruct(k), A), 1e-13);
        EXPECT_LT(max_abs_diff(f.U.transpose() * f.U, DenseMatrix<double>::identity(k)), 1e-13);
        EXPECT_LT(max_abs_diff(f.V.transpose() * f.V, DenseMatrix<double>::identity(k)), 1e-13);
    }
}

TEST(Svd, MatchesEigenvaluesOfGram) {
    const auto A = random_matrix(12, 5, 3);
    const auto f = svd(A);
    const auto e = jacobi_eigen(gram(A));
    for (std::size_t j = 0; j < 5; ++j) EXPECT_NEAR(f.s[j] * f.s[j], e.eigenvalues[j], 1e-11 * e.eigenvalues[0]);
}

TEST(Svd, KnownValuesAndConditionNumber) {
    const auto f = svd(DenseMatrix<double>(2, 2, {1, 1, 0, 1}));
    const double phi = (1 + std::sqrt(5.0)) / 2;
    EXPECT_NEAR(f.s[0], phi, 1e-15);
    EXPECT_NEAR(f.s[1], 1 / phi, 1e-15);
    EXPECT_NEAR(f.condition_number(), phi * phi, 1e-14);
    EXPECT_NEAR(f.norm_2(), phi, 1e-15);
    EXPECT_NEAR(condition_number(DenseMatrix<double>(2, 2, {1, 0, 0, 1e-6})), 1e6, 1e-6);
}

TEST(Svd, SmallSingularValuesKeepRelativeAccuracy) {
    // diag(1, 1e-10, 1e-20) hidden by orthogonal rotations on both sides would lose the
    // small values in any method based on A^T A; a graded matrix keeps them exactly here.
    const DenseMatrix<double> D(3, 3, {1, 0, 0, 0, 1e-10, 0, 0, 0, 1e-20});
    const auto f = svd(D);
    EXPECT_NEAR(f.s[2], 1e-20, 1e-34);
    EXPECT_EQ(f.rank(), 2u);      // default cut-off 3 eps * s[0] drops only 1e-20
    EXPECT_EQ(f.rank(1e-30), 3u);
}

TEST(Svd, RankDeficientMinimumNormSolve) {
    // Columns 0 and 2 equal: rank 2. The minimum-norm solution splits the weight evenly.
    DenseMatrix<double> X(4, 3, {1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1});
    const std::vector<double> y{1, 3, 5, 7};
    const auto f = svd(X);
    EXPECT_EQ(f.rank(), 2u);
    EXPECT_TRUE(std::isinf(f.condition_number()) || f.condition_number() > 1e14);
    const auto x = f.solve(y);
    EXPECT_NEAR(x[0], x[2], 1e-12);
    EXPECT_NEAR(x[0] + x[2], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
    // Pseudo-inverse satisfies the Penrose condition A A+ A = A.
    const auto P = f.pseudo_inverse();
    EXPECT_LT(max_abs_diff(X * P * X, X), 1e-12);
}

TEST(Svd, EckartYoungError) {
    const auto A = random_matrix(9, 6, 21);
    const auto f = svd(A);
    for (std::size_t k = 1; k < 6; ++k) {
        const auto E = A - f.reconstruct(k);
        double tail = 0;
        for (std::size_t j = k; j < 6; ++j) tail += f.s[j] * f.s[j];
        EXPECT_NEAR(E.norm_frobenius(), std::sqrt(tail), 1e-12);
        EXPECT_NEAR(svd(E).s[0], f.s[k], 1e-12);
    }
}

TEST(LeastSquares, MethodsAgreeOnWellConditionedProblem) {
    const auto X = random_matrix(40, 5, 5);
    std::vector<double> y(40);
    std::mt19937_64 rng(1);
    std::normal_distribution<double> z;
    for (auto& v : y) v = z(rng);
    const auto qr_fit = least_squares(X, y, LeastSquaresMethod::HouseholderQR);
    const auto ne_fit = least_squares(X, y, LeastSquaresMethod::NormalEquations);
    const auto sv_fit = least_squares(X, y, LeastSquaresMethod::SVD);
    EXPECT_LT(relative_error(ne_fit.coefficients, qr_fit.coefficients), 1e-13);
    EXPECT_LT(relative_error(sv_fit.coefficients, qr_fit.coefficients), 1e-13);
    EXPECT_NEAR(ne_fit.residual_norm, qr_fit.residual_norm, 1e-12);
    // Residual is orthogonal to the columns.
    for (std::size_t j = 0; j < 5; ++j) {
        double dot = 0;
        for (std::size_t i = 0; i < 40; ++i) dot += X(i, j) * qr_fit.residuals[i];
        EXPECT_NEAR(dot, 0.0, 1e-12);
    }
    EXPECT_EQ(to_string(LeastSquaresMethod::SVD), "SVD");
}

TEST(LeastSquares, NormalEquationsSquareTheConditionNumber) {
    // kappa(X) = 1e6 from near-collinear columns, consistent system:
    // QR/SVD errors ~ u*kappa ~ 1e-10, normal equations ~ u*kappa^2 ~ 1e-4.
    const auto X = with_singular_values(30, {1, 1e-2, 1e-4, 1e-6}, 9);
    EXPECT_NEAR(svd(X).condition_number(), 1e6, 1.0);
    const std::vector<double> beta{1, -2, 3, -4};
    const auto y = X * beta;
    const double qr_err = relative_error(least_squares(X, y).coefficients, beta);
    const double sv_err = relative_error(least_squares(X, y, LeastSquaresMethod::SVD).coefficients, beta);
    const double ne_err = relative_error(least_squares(X, y, LeastSquaresMethod::NormalEquations).coefficients, beta);
    EXPECT_LT(qr_err, 1e-8);
    EXPECT_LT(sv_err, 1e-8);
    EXPECT_GT(ne_err, 1e-7);
    EXPECT_GT(ne_err, 1000 * std::max(qr_err, sv_err));
}

TEST(LeastSquares, ColumnScalingAloneIsHarmless) {
    // kappa ~ 1e6 from column scaling only (van der Sluis): every method stays near u*kappa or better.
    const auto X = graded(30, 4, 2.0, 9);
    const std::vector<double> beta{1, -2, 3, -4};
    const auto y = X * beta;
    for (const auto m : {LeastSquaresMethod::NormalEquations, LeastSquaresMethod::HouseholderQR, LeastSquaresMethod::SVD})
        EXPECT_LT(relative_error(least_squares(X, y, m).coefficients, beta), 1e-9) << to_string(m);
}

TEST(LeastSquares, FailuresAreReported) {
    DenseMatrix<double> X(3, 2, {1, 1, 1, 1, 1, 1});
    const std::vector<double> y{1, 2, 3};
    EXPECT_THROW(least_squares(X, y, LeastSquaresMethod::HouseholderQR), std::runtime_error);
    EXPECT_THROW(least_squares(X, y, LeastSquaresMethod::NormalEquations), std::runtime_error);
    const auto fit = least_squares(X, y, LeastSquaresMethod::SVD);
    EXPECT_EQ(fit.rank, 1u);
    EXPECT_THROW(least_squares(X, std::vector<double>{1, 2}), std::invalid_argument);
}

TEST(LeastSquares, BackwardErrorOfLuIsTiny) {
    const auto A = random_matrix(50, 50, 4);
    std::vector<double> b(50, 1.0);
    const auto x = lu(A).solve(b);
    EXPECT_LT(backward_error(A, x, b), 1e-15);
}

TEST(Qr, RankToleranceIsScaleInvariant) {
    const auto A = random_matrix(6, 3, 8);
    for (const double scale : {1e-30, 1.0, 1e30}) {
        auto B = A;
        B *= scale;
        EXPECT_TRUE(qr(B).is_full_rank()) << scale;
        const auto Ri = qr(B).r_inverse();
        const auto R  = qr(B).R;
        DenseMatrix<double> R3(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) R3(i, j) = R(i, j);
        EXPECT_LT(max_abs_diff(R3 * Ri, DenseMatrix<double>::identity(3)), 1e-13);
    }
}

TEST(JacobiEigen, ScaleInvariant) {
    // Before the fix the absolute stopping test returned diag(1e-7, 1e-7) here.
    const auto e = jacobi_eigen(DenseMatrix<double>(2, 2, {1e-7, 1e-7, 1e-7, 1e-7}));
    EXPECT_NEAR(e.eigenvalues[0], 2e-7, 1e-20);
    EXPECT_NEAR(e.eigenvalues[1], 0.0, 1e-20);
    const auto big = jacobi_eigen(DenseMatrix<double>(2, 2, {2e9, 1e9, 1e9, 2e9}));
    EXPECT_NEAR(big.eigenvalues[0], 3e9, 1e-5);
    EXPECT_NEAR(big.eigenvalues[1], 1e9, 1e-5);
}

TEST(Lu, EmptyMatrix) {
    const auto f = lu(DenseMatrix<double>(0, 0));
    EXPECT_FALSE(f.singular);
    EXPECT_EQ(f.det(), 1.0);
}

TEST(LinearRegression, CovarianceSurvivesIllConditioning) {
    // kappa(X) = 1e9: X^T X has kappa ~ 1e18, so its Cholesky inverse (the old method) is garbage.
    const auto X = with_singular_values(40, {1, 1e-3, 1e-6, 1e-9}, 2);
    // (Cholesky of X^T X may or may not break down; its inverse is inaccurate either way.)
    std::vector<double> y(40);
    std::mt19937_64 rng(3);
    std::normal_distribution<double> z;
    for (std::size_t i = 0; i < 40; ++i) y[i] = 1 + z(rng);
    const auto fit = linear_regression(X, y);
    // Covariance must match sigma^2 (X^T X)^{-1} computed from the SVD.
    const auto f = svd(X);
    for (std::size_t a = 0; a < 4; ++a) {
        double v = 0;
        for (std::size_t k = 0; k < 4; ++k) v += f.V(a, k) * f.V(a, k) / (f.s[k] * f.s[k]);
        EXPECT_NEAR(fit.covariance(a, a) / (fit.sigma2 * v), 1.0, 1e-5);
    }
}

TEST(LeastSquares, VandermondeAndCast) {
    const auto V = vandermonde(std::vector<double>{2, 3}, 3);
    EXPECT_EQ(V.cols(), 4u);
    EXPECT_EQ(V(0, 3), 8.0);
    EXPECT_EQ(V(1, 2), 9.0);
    const auto L = matrix_cast<long double>(V);
    EXPECT_EQ(L(1, 3), 27.0L);
}

TEST(LeastSquares, AccuracyAuditShowsKappaSquared) {
    std::vector<double> x(30), y(30);
    for (std::size_t i = 0; i < 30; ++i) {
        x[i] = 1 + 0.1 * static_cast<double>(i);
        y[i] = std::sin(x[i]);
    }
    const auto a = least_squares_accuracy(vandermonde(x, 6), y);
    EXPECT_GT(a.condition_number, a.equilibrated_condition_number);
    EXPECT_GT(a.equilibrated_condition_number, 1e4);
    EXPECT_LT(a.tan_theta, 1e-3);  // sin is nearly a degree-6 polynomial here
    EXPECT_LT(a.qr_error, 100 * 1.1e-16 * a.equilibrated_condition_number);
    EXPECT_GT(a.normal_equations_error, 100 * a.qr_error);
    EXPECT_EQ(a.reference.size(), 7u);
}

TEST(LeastSquares, ColumnHelpers) {
    const auto X = random_matrix(20, 3, 17);
    const auto E = equilibrate_columns(X);
    for (std::size_t j = 0; j < 3; ++j) EXPECT_NEAR(norm_of_column(E, j), 1.0, 1e-15);
    const auto R = correlation_matrix(X);
    for (std::size_t j = 0; j < 3; ++j) EXPECT_NEAR(R(j, j), 1.0, 1e-14);
    EXPECT_NEAR(R(0, 1), R(1, 0), 0.0);
    EXPECT_LE(std::abs(R(0, 2)), 1.0);
}
