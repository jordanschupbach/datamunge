#include <gtest/gtest.h>

#include <datamunge/algorithms/linear_algebra.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

using datamunge::algorithms::EigenPair;
using datamunge::algorithms::freivalds_verify;
using datamunge::algorithms::gram_schmidt;
using datamunge::algorithms::power_iteration;
using datamunge::algorithms::strassen_multiply;

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace {

Matrix naive_mul(const Matrix& A, const Matrix& B) {
    const std::size_t m = A.size(), k = A[0].size(), n = B[0].size();
    Matrix            C(m, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p)
            for (std::size_t j = 0; j < n; ++j) C[i][j] += A[i][p] * B[p][j];
    return C;
}
double dot(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
Matrix random_matrix(std::mt19937& rng, std::size_t r, std::size_t c) {
    std::uniform_int_distribution<int> v(-5, 5);
    Matrix                             M(r, Vector(c));
    for (auto& row : M)
        for (double& x : row) x = v(rng);
    return M;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Gram-Schmidt
// ------------------------------------------------------------------------------------------------

TEST(GramSchmidt, OrthonormalAndSpanPreserved) {
    std::mt19937 rng(1);
    for (int t = 0; t < 500; ++t) {
        const Matrix V = random_matrix(rng, 4, 6); // 4 vectors in R^6 (a.s. independent)
        const Matrix Q = gram_schmidt(V);

        for (std::size_t i = 0; i < Q.size(); ++i)
            for (std::size_t j = 0; j < Q.size(); ++j)
                EXPECT_NEAR(dot(Q[i], Q[j]), i == j ? 1.0 : 0.0, 1e-9); // orthonormal

        // Every input lies in the span of Q: its residual after projection vanishes.
        for (const Vector& v : V) {
            Vector res = v;
            for (const Vector& q : Q) {
                const double p = dot(res, q);
                for (std::size_t i = 0; i < res.size(); ++i) res[i] -= p * q[i];
            }
            EXPECT_NEAR(std::sqrt(dot(res, res)), 0.0, 1e-9);
        }
    }
}

TEST(GramSchmidt, DropsDependentVectors) {
    const Matrix V = {{1, 0, 0}, {2, 0, 0}, {0, 1, 0}}; // second is dependent on the first
    const Matrix Q = gram_schmidt(V);
    EXPECT_EQ(Q.size(), 2u);
}

// ------------------------------------------------------------------------------------------------
// Power iteration
// ------------------------------------------------------------------------------------------------

TEST(PowerIteration, KnownEigenpairs) {
    const EigenPair a = power_iteration({{3, 1}, {1, 3}}); // eigenvalues 4 (v=(1,1)) and 2
    EXPECT_NEAR(std::fabs(a.value), 4.0, 1e-6);
    EXPECT_NEAR(std::fabs(a.vector[0]), std::fabs(a.vector[1]), 1e-6);

    const EigenPair b = power_iteration({{2, 0}, {0, -3}}); // largest magnitude is -3
    EXPECT_NEAR(b.value, -3.0, 1e-6);
    EXPECT_NEAR(std::fabs(b.vector[1]), 1.0, 1e-6);
}

TEST(PowerIteration, SatisfiesEigenEquation) {
    std::mt19937                           rng(7);
    std::uniform_real_distribution<double> u(1.0, 5.0);
    for (int t = 0; t < 300; ++t) {
        // Diagonal (hence symmetric) matrix with a clearly dominant entry (10 vs <= 5), so power
        // iteration converges to the corresponding basis eigenvector.
        const std::size_t n = 5;
        const std::size_t d = static_cast<std::size_t>(rng()) % n; // which index dominates
        Matrix            A(n, Vector(n, 0.0));
        for (std::size_t i = 0; i < n; ++i) A[i][i] = (i == d) ? 10.0 : u(rng);
        const EigenPair e = power_iteration(A);
        EXPECT_NEAR(e.value, 10.0, 1e-6);
        EXPECT_NEAR(std::fabs(e.vector[d]), 1.0, 1e-6); // eigenvector is (near) the d-th basis vector
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(A[i][i] * e.vector[i], e.value * e.vector[i], 1e-6); // A v == lambda v
    }
}

// ------------------------------------------------------------------------------------------------
// Strassen
// ------------------------------------------------------------------------------------------------

TEST(Strassen, MatchesNaiveSquareAndRectangular) {
    std::mt19937 rng(11);
    for (auto [m, k, n] : std::vector<std::array<std::size_t, 3>>{
             {8, 8, 8}, {96, 96, 96}, {130, 130, 130}, {40, 55, 30}, {70, 20, 90}}) {
        const Matrix A = random_matrix(rng, m, k);
        const Matrix B = random_matrix(rng, k, n);
        EXPECT_EQ(strassen_multiply(A, B), naive_mul(A, B)) << m << "x" << k << " * " << k << "x" << n;
    }
}

// ------------------------------------------------------------------------------------------------
// Freivalds
// ------------------------------------------------------------------------------------------------

TEST(Freivalds, AcceptsCorrectRejectsWrong) {
    std::mt19937 rng(13);
    for (int t = 0; t < 2000; ++t) {
        const Matrix A = random_matrix(rng, 6, 5);
        const Matrix B = random_matrix(rng, 5, 7);
        Matrix       C = naive_mul(A, B);
        EXPECT_TRUE(freivalds_verify(A, B, C)); // correct product always accepted

        Matrix wrong = C;
        wrong[static_cast<std::size_t>(rng()) % 6][static_cast<std::size_t>(rng()) % 7] += 1.0;
        EXPECT_FALSE(freivalds_verify(A, B, wrong)); // one wrong entry caught w.h.p. over 20 rounds
    }
}
