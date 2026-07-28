#include <gtest/gtest.h>

#include <datamunge/algorithms/linear_solvers.hpp>

#include <cmath>
#include <random>
#include <vector>

using datamunge::algorithms::biconjugate_gradient;
using datamunge::algorithms::gauss_jordan;
using datamunge::algorithms::levinson_solve;
using datamunge::algorithms::sor;
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
        A[i][i] = off + 1.0 + std::fabs(u(rng));
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

TEST(GaussJordan, SolvesAndInverts) {
    const auto s = gauss_jordan({{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}, {8, -11, -3});
    ASSERT_TRUE(s.solved);
    EXPECT_NEAR(s.x[0], 2.0, 1e-9);
    EXPECT_NEAR(s.x[1], 3.0, 1e-9);
    EXPECT_NEAR(s.x[2], -1.0, 1e-9);

    std::mt19937 rng(1);
    for (int t = 0; t < 800; ++t) {
        const std::size_t n = 6;
        const Matrix      A = diagonally_dominant(rng, n);
        const Vector      b = random_vector(rng, n);
        const auto        r = gauss_jordan(A, b);
        ASSERT_TRUE(r.solved);
        EXPECT_LT(residual(A, r.x, b), 1e-8);
        // A * A^{-1} == I.
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double s2 = 0.0;
                for (std::size_t k = 0; k < n; ++k) s2 += A[i][k] * r.inverse[k][j];
                EXPECT_NEAR(s2, i == j ? 1.0 : 0.0, 1e-8);
            }
    }
    EXPECT_FALSE(gauss_jordan({{1, 2}, {2, 4}}, {1, 2}).solved); // singular
}

TEST(SOR, ConvergesAndBeatsGaussSeidel) {
    std::mt19937 rng(2);
    for (int t = 0; t < 500; ++t) {
        const std::size_t n = 8;
        const Matrix      A = diagonally_dominant(rng, n);
        const Vector      b = random_vector(rng, n);
        const auto        r = sor(A, b, 1.2);
        EXPECT_TRUE(r.converged);
        EXPECT_LT(residual(A, r.x, b), 1e-8);
    }
    // On the SPD 1-D Laplacian, an over-relaxed omega converges faster than plain Gauss-Seidel.
    const std::size_t n = 30;
    Matrix            L(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) { L[i][i] = 2.0; if (i) L[i][i - 1] = -1.0; if (i + 1 < n) L[i][i + 1] = -1.0; }
    const Vector b(n, 1.0);
    const int    gs_iters  = sor(L, b, 1.0, 100000, 1e-8).iterations; // omega=1 is Gauss-Seidel
    const int    sor_iters = sor(L, b, 1.8, 100000, 1e-8).iterations;
    EXPECT_LT(sor_iters, gs_iters);
}

TEST(BiconjugateGradient, SolvesNonsymmetricSystems) {
    std::mt19937 rng(3);
    for (int t = 0; t < 500; ++t) {
        const std::size_t n = 10;
        const Matrix      A = diagonally_dominant(rng, n); // non-symmetric in general
        const Vector      b = random_vector(rng, n);
        const auto        r = biconjugate_gradient(A, b, 4 * static_cast<int>(n));
        EXPECT_TRUE(r.converged);
        EXPECT_LT(residual(A, r.x, b), 1e-7);
    }
}

TEST(Levinson, MatchesDenseSolveOnToeplitz) {
    std::mt19937                           rng(4);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    for (int t = 0; t < 2000; ++t) {
        const std::size_t n = 8;
        Vector            row(n);
        double            off = 0.0;
        for (std::size_t k = 1; k < n; ++k) { row[k] = u(rng); off += std::fabs(row[k]); }
        row[0] = off + 1.0 + std::fabs(u(rng)); // diagonally dominant -> nonsingular symmetric Toeplitz
        const Vector b = random_vector(rng, n);

        Matrix T(n, Vector(n));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) T[i][j] = row[i > j ? i - j : j - i];

        const Vector x = levinson_solve(row, b);
        EXPECT_LT(residual(T, x, b), 1e-7);
    }
}
