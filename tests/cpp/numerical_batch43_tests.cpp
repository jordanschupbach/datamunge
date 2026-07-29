#include <gtest/gtest.h>

#include <datamunge/algorithms/simplex.hpp>
#include <datamunge/algorithms/stone_sip.hpp>
#include <datamunge/algorithms/wang_landau.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Simplex ----------------

TEST(Simplex, SolvesKnownLps) {
    // max 3x+2y s.t. x+y<=4, x+3y<=6  ->  (4,0), value 12
    auto r1 = simplex_maximize({3, 2}, {{1, 1}, {1, 3}}, {4, 6});
    EXPECT_EQ(r1.status, SimplexStatus::Optimal);
    EXPECT_NEAR(r1.value, 12.0, 1e-9);
    EXPECT_NEAR(r1.x[0], 4.0, 1e-9);
    EXPECT_NEAR(r1.x[1], 0.0, 1e-9);

    // classic: max 5x+4y s.t. 6x+4y<=24, x+2y<=6  ->  (3, 1.5), value 21
    auto r2 = simplex_maximize({5, 4}, {{6, 4}, {1, 2}}, {24, 6});
    EXPECT_NEAR(r2.value, 21.0, 1e-9);
    EXPECT_NEAR(r2.x[0], 3.0, 1e-9);
    EXPECT_NEAR(r2.x[1], 1.5, 1e-9);
}

TEST(Simplex, DetectsUnbounded) {
    // max x s.t. x - y <= 1 (x can grow with y)  -> unbounded
    auto r = simplex_maximize({1, 0}, {{1, -1}}, {1});
    EXPECT_EQ(r.status, SimplexStatus::Unbounded);
}

TEST(Simplex, FeasibleAndOptimalOnRandomLps) {
    std::mt19937_64                        rng(5);
    std::uniform_real_distribution<double> coef(0.1, 3.0);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 3, m = 4;
        std::vector<double> c(n);
        for (auto& v : c) v = coef(rng);
        std::vector<std::vector<double>> A(m, std::vector<double>(n));
        std::vector<double>              b(m);
        for (std::size_t i = 0; i < m; ++i) { for (auto& a : A[i]) a = coef(rng); b[i] = coef(rng) * 5; }
        const auto r = simplex_maximize(c, A, b);
        ASSERT_EQ(r.status, SimplexStatus::Optimal) << "trial " << trial;
        // feasibility of the returned vertex
        for (std::size_t i = 0; i < m; ++i) {
            double lhs = 0; for (std::size_t j = 0; j < n; ++j) lhs += A[i][j] * r.x[j];
            EXPECT_LE(lhs, b[i] + 1e-6) << "trial " << trial << " constraint " << i;
        }
        for (double xj : r.x) EXPECT_GE(xj, -1e-9);
        // objective consistency
        double obj = 0; for (std::size_t j = 0; j < n; ++j) obj += c[j] * r.x[j];
        EXPECT_NEAR(obj, r.value, 1e-6) << "trial " << trial;
    }
}

// ---------------- Wang-Landau ----------------

TEST(WangLandau, RecoversBinomialDensityOfStates) {
    const int  N    = 24;
    const auto logg = wang_landau_dos(N, 0.85, 1e-6, 1);
    ASSERT_EQ(logg.size(), static_cast<std::size_t>(N + 1));
    EXPECT_NEAR(logg[0], 0.0, 1e-12); // normalized
    double maxerr = 0;
    for (int E = 0; E <= N; ++E) {
        const double exact = std::lgamma(N + 1) - std::lgamma(E + 1) - std::lgamma(N - E + 1);
        maxerr = std::max(maxerr, std::fabs(logg[E] - exact));
    }
    EXPECT_LT(maxerr, 0.5) << "max abs error " << maxerr;
}

// ---------------- Stone's method (SIP) ----------------

TEST(StoneSip, SolvesPoissonSystem) {
    const int Nx = 20, Ny = 20, M = Nx * Ny;
    std::vector<double> Ap(M, 4), Aw(M, 0), Ae(M, 0), As(M, 0), An(M, 0);
    auto idx = [Nx](int i, int j) { return j * Nx + i; };
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            const int p = idx(i, j);
            if (i > 0) Aw[p] = -1; if (i < Nx - 1) Ae[p] = -1;
            if (j > 0) As[p] = -1; if (j < Ny - 1) An[p] = -1;
        }
    std::mt19937_64                        rng(1);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double>                    xt(M);
    for (auto& v : xt) v = u(rng);
    std::vector<double> b(M, 0);
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            const int p = idx(i, j);
            double    ax = Ap[p] * xt[p];
            if (i > 0) ax += Aw[p] * xt[idx(i - 1, j)];
            if (i < Nx - 1) ax += Ae[p] * xt[idx(i + 1, j)];
            if (j > 0) ax += As[p] * xt[idx(i, j - 1)];
            if (j < Ny - 1) ax += An[p] * xt[idx(i, j + 1)];
            b[p] = ax;
        }
    const auto res = stone_sip(Nx, Ny, Ap, Aw, Ae, As, An, b, 0.92, 500, 1e-11);
    EXPECT_LT(res.residual, 1e-10);
    double maxerr = 0;
    for (int p = 0; p < M; ++p) maxerr = std::max(maxerr, std::fabs(res.x[p] - xt[p]));
    EXPECT_LT(maxerr, 1e-9);
}

TEST(StoneSip, ResidualDecreasesMonotonically) {
    const int Nx = 12, Ny = 12, M = Nx * Ny;
    std::vector<double> Ap(M, 4), Aw(M, 0), Ae(M, 0), As(M, 0), An(M, 0);
    auto idx = [Nx](int i, int j) { return j * Nx + i; };
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            const int p = idx(i, j);
            if (i > 0) Aw[p] = -1; if (i < Nx - 1) Ae[p] = -1;
            if (j > 0) As[p] = -1; if (j < Ny - 1) An[p] = -1;
        }
    std::vector<double> b(M, 1.0); // unit source
    double prev = 1e300;
    for (int it = 1; it <= 15; ++it) {
        const auto r = stone_sip(Nx, Ny, Ap, Aw, Ae, As, An, b, 0.92, it, 1e-15);
        EXPECT_LE(r.residual, prev + 1e-12) << "iter " << it;
        prev = r.residual;
    }
}
