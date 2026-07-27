#include <gtest/gtest.h>

#include <datamunge/algorithms/linear_solvers.hpp>

#include <cmath>
#include <random>
#include <vector>

using datamunge::algorithms::conjugate_gradient;
using datamunge::algorithms::gauss_seidel;
using datamunge::algorithms::gaussian_elimination;
using datamunge::algorithms::thomas_solve;
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace {

double residual(const Matrix& A, const Vector& x, const Vector& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        double ax = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) ax += A[i][j] * x[j];
        s += (ax - b[i]) * (ax - b[i]);
    }
    return std::sqrt(s);
}

Matrix diagonally_dominant(std::mt19937& rng, std::size_t n) {
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    Matrix                                 A(n, Vector(n));
    for (std::size_t i = 0; i < n; ++i) {
        double off = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            if (i != j) { A[i][j] = u(rng); off += std::fabs(A[i][j]); }
        A[i][i] = off + 1.0 + std::fabs(u(rng)); // strictly diagonally dominant
    }
    return A;
}

Matrix spd(std::mt19937& rng, std::size_t n) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Matrix                                 M(n, Vector(n));
    for (auto& r : M)
        for (double& x : r) x = u(rng);
    Matrix A(n, Vector(n, 0.0)); // A = M^T M + n I  (symmetric positive-definite)
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) A[i][j] += M[k][i] * M[k][j];
            if (i == j) A[i][j] += static_cast<double>(n);
        }
    return A;
}

Vector random_vector(std::mt19937& rng, std::size_t n) {
    std::uniform_real_distribution<double> u(-5.0, 5.0);
    Vector                                 v(n);
    for (double& x : v) x = u(rng);
    return v;
}

} // namespace

TEST(GaussianElimination, KnownAndResidual) {
    const auto s = gaussian_elimination({{2, 1}, {1, 3}}, {3, 5});
    ASSERT_TRUE(s.solved);
    EXPECT_NEAR(s.x[0], 0.8, 1e-12);
    EXPECT_NEAR(s.x[1], 1.4, 1e-12);

    std::mt19937 rng(1);
    for (int t = 0; t < 1000; ++t) {
        const std::size_t n = 6;
        const Matrix      A = diagonally_dominant(rng, n);
        const Vector      b = random_vector(rng, n);
        const auto        r = gaussian_elimination(A, b);
        ASSERT_TRUE(r.solved);
        EXPECT_LT(residual(A, r.x, b), 1e-8);
    }
}

TEST(GaussianElimination, DetectsSingular) {
    const auto s = gaussian_elimination({{1, 2}, {2, 4}}, {1, 2}); // second row = 2 * first
    EXPECT_FALSE(s.solved);
}

TEST(GaussSeidel, ConvergesOnDiagonallyDominant) {
    std::mt19937 rng(2);
    for (int t = 0; t < 500; ++t) {
        const std::size_t n = 8;
        const Matrix      A = diagonally_dominant(rng, n);
        const Vector      b = random_vector(rng, n);
        const auto        r = gauss_seidel(A, b);
        EXPECT_TRUE(r.converged);
        EXPECT_LT(r.residual, 1e-9);
        EXPECT_LT(residual(A, r.x, b), 1e-8);
    }
}

TEST(ConjugateGradient, SolvesSPDInAtMostNSteps) {
    std::mt19937 rng(3);
    for (int t = 0; t < 500; ++t) {
        const std::size_t n = 10;
        const Matrix      A = spd(rng, n);
        const Vector      b = random_vector(rng, n);
        const auto        r = conjugate_gradient(A, b, 2 * static_cast<int>(n));
        EXPECT_TRUE(r.converged);
        EXPECT_LT(residual(A, r.x, b), 1e-7);
        EXPECT_LE(r.iterations, static_cast<int>(n) + 2); // exact in <= n steps (a little slack for rounding)
    }
}

TEST(Thomas, MatchesDenseSolve) {
    std::mt19937                           rng(4);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    for (int t = 0; t < 2000; ++t) {
        const std::size_t n = 12;
        Vector sub(n), diag(n), super(n), rhs = random_vector(rng, n);
        for (std::size_t i = 0; i < n; ++i) {
            sub[i]   = (i == 0) ? 0.0 : u(rng);
            super[i] = (i + 1 == n) ? 0.0 : u(rng);
            diag[i]  = std::fabs(sub[i]) + std::fabs(super[i]) + 1.0 + std::fabs(u(rng)); // dominant
        }
        // Dense version of the tridiagonal matrix.
        Matrix A(n, Vector(n, 0.0));
        for (std::size_t i = 0; i < n; ++i) {
            A[i][i] = diag[i];
            if (i > 0) A[i][i - 1] = sub[i];
            if (i + 1 < n) A[i][i + 1] = super[i];
        }
        const Vector x = thomas_solve(sub, diag, super, rhs);
        EXPECT_LT(residual(A, x, rhs), 1e-8);
    }
}
