#include <gtest/gtest.h>

#include <datamunge/algorithms/linear_solvers.hpp>
#include <datamunge/algorithms/network_analysis.hpp>

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

TEST(BiCGOperator, MatchesDenseBiCGAndSolvesNonSymmetricSystems) {
    std::mt19937 rng(21);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const std::size_t n = 40;
    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) a[i][j] = (i == j ? 8.0 : 0.0) + u(rng);
    std::vector<double> b(n);
    for (auto& v : b) v = u(rng);
    const auto mul = [&](bool transpose) {
        return [&, transpose](const std::vector<double>& v) {
            std::vector<double> y(n, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) y[i] += (transpose ? a[j][i] : a[i][j]) * v[j];
            return y;
        };
    };
    const auto run = datamunge::algorithms::biconjugate_gradient(mul(false), mul(true), b, 200, 1e-12);
    ASSERT_TRUE(run.converged);
    EXPECT_FALSE(run.breakdown);
    EXPECT_EQ(run.products, 2 * run.iterations);
    EXPECT_EQ(run.residual_history.size(), static_cast<std::size_t>(run.iterations) + 1);
    const auto ax = mul(false)(run.x);
    for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(ax[i], b[i], 1e-9);
    const auto dense = datamunge::algorithms::biconjugate_gradient(a, b, 200, 1e-14);
    for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(dense.x[i], run.x[i], 1e-9);
}

TEST(BiCGOperator, BreaksDownOnASwapMatrix) {
    // A = [[0, 1], [1, 0]] is orthogonal, but p~^T A p = 0 at the first step for b = e1.
    const auto swap = [](const std::vector<double>& v) { return std::vector<double>{v[1], v[0]}; };
    const auto run = datamunge::algorithms::biconjugate_gradient(swap, swap, {1.0, 0.0}, 10, 1e-12);
    EXPECT_TRUE(run.breakdown);
    EXPECT_FALSE(run.converged);
    EXPECT_EQ(run.iterations, 0);
}

TEST(BiCGOperator, LeftEigenvectorShadowIsASeriousBreakdownAndARandomShadowFixesIt) {
    // A = I - alpha P^T for a row-stochastic P: 1^T A = (1 - alpha) 1^T, so the shadow b = 1/n
    // spans an A^T-invariant line and r~_1 = 0 exactly.
    const std::size_t n = 30;
    const double alpha = 0.9;
    std::mt19937 rng(2);
    std::vector<std::vector<std::size_t>> out(n);
    for (std::size_t i = 0; i < n; ++i)
        for (int e = 0; e < 3; ++e) out[i].push_back(rng() % n);
    const auto A = [&](const std::vector<double>& v) {
        std::vector<double> y = v;
        for (std::size_t i = 0; i < n; ++i)
            for (const auto j : out[i]) y[j] -= alpha / 3.0 * v[i];
        return y;
    };
    const auto At = [&](const std::vector<double>& v) {
        std::vector<double> y = v;
        for (std::size_t i = 0; i < n; ++i)
            for (const auto j : out[i]) y[i] -= alpha / 3.0 * v[j];
        return y;
    };
    const std::vector<double> b(n, (1 - alpha) / n);
    const auto stuck = datamunge::algorithms::biconjugate_gradient(A, At, b, 200, 1e-12);
    EXPECT_TRUE(stuck.breakdown);
    EXPECT_EQ(stuck.iterations, 1);
    std::vector<double> shadow(n);
    std::uniform_real_distribution<double> u(0.5, 1.5);
    for (auto& v : shadow) v = u(rng);
    const auto run = datamunge::algorithms::biconjugate_gradient(A, At, b, 200, 1e-12, shadow);
    EXPECT_TRUE(run.converged) << run.iterations << " " << run.breakdown << " " << run.residual_history.back();
    EXPECT_THROW((void)datamunge::algorithms::biconjugate_gradient(A, At, b, 10, 1e-12, {1.0}), std::invalid_argument);
}

TEST(PageRankSystem, SolvingTheSystemReproducesPageRank) {
    std::mt19937 rng(8);
    const std::size_t n = 60;
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t u = 0; u < n; ++u)
        for (int e = 0; e < 4; ++e) edges.emplace_back(u, rng() % n);
    const auto sys = datamunge::algorithms::page_rank_system(n, edges, 0.85);
    const auto power = datamunge::algorithms::page_rank(n, edges, 0.85, 1e-15, 10000);
    std::vector<double> shadow(n);
    for (auto& v : shadow) v = 0.5 + (rng() % 1000) / 1000.0;
    const auto run = datamunge::algorithms::biconjugate_gradient(
        [&](const std::vector<double>& v) { return sys.matrix.spmv(v); },
        [&](const std::vector<double>& v) { return sys.transpose.spmv(v); }, sys.rhs, 500, 1e-13, shadow);
    ASSERT_TRUE(run.converged);
    double total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(run.x[i], power[i], 1e-10);
        total += run.x[i];
    }
    EXPECT_NEAR(total, 1.0, 1e-10);
    EXPECT_THROW((void)datamunge::algorithms::page_rank_system(3, {{0, 1}, {1, 2}}, 0.85), std::invalid_argument);
}
