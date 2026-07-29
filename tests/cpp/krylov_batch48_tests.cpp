#include <gtest/gtest.h>

#include <datamunge/algorithms/arnoldi_iteration.hpp>
#include <datamunge/algorithms/lanczos_iteration.hpp>
#include <datamunge/algorithms/nearest_neighbor_interpolation.hpp>
#include <datamunge/algorithms/qr_algorithm.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {
using Mat = std::vector<std::vector<double>>;

Mat random_matrix(std::mt19937& rng, int n, bool symmetric) {
    std::normal_distribution<double> g(0, 1);
    Mat A(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) A[i][j] = g(rng);
    if (symmetric)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) A[i][j] = A[j][i];
    return A;
}
} // namespace

// ---------------- Arnoldi iteration ----------------

TEST(ArnoldiIteration, SatisfiesArnoldiRelationAndOrthonormality) {
    std::mt19937 rng(1);
    double       max_rel = 0, max_orth = 0;
    for (int t = 0; t < 200; ++t) {
        const int n = 6 + static_cast<int>(rng() % 4);
        const Mat A = random_matrix(rng, n, false);
        std::vector<double> b(n);
        for (auto& x : b) x = std::normal_distribution<double>(0, 1)(rng);
        const auto r  = arnoldi_iteration(A, b, n);
        const int  mm = static_cast<int>(r.V.size()) - 1;
        // Arnoldi relation: A v_j = sum_i H[i][j] v_i.
        for (int j = 0; j < mm; ++j) {
            std::vector<double> Av(n, 0);
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < n; ++k) Av[i] += A[i][k] * r.V[j][k];
            for (int i = 0; i < static_cast<int>(r.V.size()); ++i)
                for (int k = 0; k < n; ++k) Av[k] -= r.H[i][j] * r.V[i][k];
            double e = 0; for (double v : Av) e += v * v;
            max_rel = std::max(max_rel, std::sqrt(e));
        }
        // Orthonormality of the basis.
        for (std::size_t a = 0; a < r.V.size(); ++a)
            for (std::size_t c = 0; c < r.V.size(); ++c) {
                double d = 0; for (int k = 0; k < n; ++k) d += r.V[a][k] * r.V[c][k];
                max_orth = std::max(max_orth, std::fabs(d - (a == c ? 1.0 : 0.0)));
            }
    }
    EXPECT_LT(max_rel, 1e-9);
    EXPECT_LT(max_orth, 1e-9);
}

// ---------------- Lanczos iteration ----------------

TEST(LanczosIteration, FullTridiagonalRecoversSpectrum) {
    std::mt19937 rng(2);
    double       max_err = 0;
    for (int t = 0; t < 200; ++t) {
        const int n = 5 + static_cast<int>(rng() % 4);
        const Mat A = random_matrix(rng, n, true);
        std::vector<double> b(n);
        for (auto& x : b) x = std::normal_distribution<double>(0, 1)(rng);
        const auto r  = lanczos_iteration(A, b, n);
        const int  mm = static_cast<int>(r.alpha.size());
        if (mm != n) continue; // early termination (rare) -> skip
        Mat T(mm, std::vector<double>(mm, 0));
        for (int i = 0; i < mm; ++i) {
            T[i][i] = r.alpha[i];
            if (i + 1 < mm) { T[i][i + 1] = r.beta[i]; T[i + 1][i] = r.beta[i]; }
        }
        const auto ritz = qr_eigenvalues(T);
        const auto eig  = qr_eigenvalues(A);
        for (int i = 0; i < n; ++i) max_err = std::max(max_err, std::fabs(ritz[i] - eig[i]));
    }
    EXPECT_LT(max_err, 1e-8);
}

TEST(LanczosIteration, RitzValuesApproachExtremeEigenvalues) {
    // A symmetric matrix with a wide spectrum; a few Lanczos steps should bracket
    // the largest and smallest eigenvalues closely.
    std::mt19937 rng(7);
    const int    n = 30;
    const Mat    A = random_matrix(rng, n, true);
    const auto   eig = qr_eigenvalues(A); // ascending
    std::vector<double> b(n, 1.0);
    const auto   r = lanczos_iteration(A, b, 20); // 20 of 30 steps: extremes converge to ~1e-8
    const int    m = static_cast<int>(r.alpha.size());
    Mat T(m, std::vector<double>(m, 0));
    for (int i = 0; i < m; ++i) { T[i][i] = r.alpha[i]; if (i + 1 < m) { T[i][i + 1] = r.beta[i]; T[i + 1][i] = r.beta[i]; } }
    const auto ritz = qr_eigenvalues(T); // ascending
    // Ritz values are bounded by the true spectrum and the extremes converge fast.
    EXPECT_LE(ritz.back(), eig.back() + 1e-9);
    EXPECT_GE(ritz.front(), eig.front() - 1e-9);
    EXPECT_NEAR(ritz.back(), eig.back(), 1e-5);
    EXPECT_NEAR(ritz.front(), eig.front(), 1e-5);
}

// ---------------- Nearest-neighbor interpolation ----------------

TEST(NearestNeighborInterpolation, NodesBetweenAndClamp) {
    std::vector<double> xs{0, 1, 2, 3, 4}, ys{10, 20, 30, 40, 50};
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, xs[i]), ys[i]);
    EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, 1.4), 20); // nearest node 1
    EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, 1.6), 30); // nearest node 2
    EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, 1.5), 20); // tie -> lower index
    EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, -100), 10); // clamp low
    EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, 100), 50);  // clamp high
}

TEST(NearestNeighborInterpolation, MatchesBruteForceNearest) {
    std::mt19937                           rng(3);
    std::uniform_real_distribution<double> u(0, 100);
    std::vector<double> xs, ys;
    for (int i = 0; i < 40; ++i) { xs.push_back(i * 2.5); ys.push_back(u(rng)); }
    for (int t = 0; t < 2000; ++t) {
        const double x = u(rng);
        // brute force nearest
        std::size_t best = 0; double bd = 1e300;
        for (std::size_t i = 0; i < xs.size(); ++i) { const double d = std::fabs(xs[i] - x); if (d < bd - 1e-12) { bd = d; best = i; } }
        EXPECT_EQ(nearest_neighbor_interpolate(xs, ys, x), ys[best]) << "x=" << x;
    }
}
