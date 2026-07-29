#include <gtest/gtest.h>

#include <datamunge/algorithms/inverse_iteration.hpp>
#include <datamunge/algorithms/qr_algorithm.hpp>
#include <datamunge/algorithms/rayleigh_quotient_iteration.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {
using Mat = std::vector<std::vector<double>>;

// Build a symmetric matrix A = Q diag(d) Q^T with a random orthogonal Q;
// returns A and stores the eigenvectors (columns of Q).
Mat build_symmetric(std::mt19937& rng, const std::vector<double>& d, Mat& Q) {
    const int n = static_cast<int>(d.size());
    std::normal_distribution<double> g(0, 1);
    Q.assign(n, std::vector<double>(n));
    for (int j = 0; j < n; ++j) {
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = g(rng);
        for (int k = 0; k < j; ++k) {
            double dot = 0;
            for (int i = 0; i < n; ++i) dot += Q[i][k] * v[i];
            for (int i = 0; i < n; ++i) v[i] -= dot * Q[i][k];
        }
        double nrm = 0; for (double x : v) nrm += x * x; nrm = std::sqrt(nrm);
        for (int i = 0; i < n; ++i) Q[i][j] = v[i] / nrm;
    }
    Mat A(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < n; ++k) s += Q[i][k] * d[k] * Q[j][k];
            A[i][j] = s;
        }
    return A;
}

double residual(const Mat& A, double lam, const std::vector<double>& v) {
    const int n = static_cast<int>(A.size());
    double r = 0;
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += A[i][j] * v[j];
        const double e = s - lam * v[i];
        r += e * e;
    }
    return std::sqrt(r);
}
} // namespace

// ---------------- QR algorithm ----------------

TEST(QrAlgorithm, RecoversKnownSpectrum) {
    std::mt19937 rng(1);
    double       maxerr = 0;
    for (int t = 0; t < 400; ++t) {
        const int n = 2 + static_cast<int>(rng() % 5);
        std::vector<double> d(n);
        std::uniform_real_distribution<double> u(-10, 10);
        for (auto& x : d) x = u(rng);
        Mat Q;
        const Mat A = build_symmetric(rng, d, Q);
        auto e = qr_eigenvalues(A);
        auto ds = d; std::sort(ds.begin(), ds.end());
        ASSERT_EQ(e.size(), ds.size());
        for (int i = 0; i < n; ++i) maxerr = std::max(maxerr, std::fabs(e[i] - ds[i]));
    }
    EXPECT_LT(maxerr, 1e-8);
}

// ---------------- Inverse iteration ----------------

TEST(InverseIteration, RecoversEigenpairNearestMu) {
    std::mt19937 rng(2);
    for (int t = 0; t < 300; ++t) {
        const int n = 3 + static_cast<int>(rng() % 4);
        // well-separated eigenvalues so "nearest to mu" is unambiguous
        std::vector<double> d(n);
        for (int i = 0; i < n; ++i) d[i] = -10.0 + 5.0 * i;
        Mat Q;
        const Mat A = build_symmetric(rng, d, Q);
        const int  tgt = static_cast<int>(rng() % n);
        const auto p   = inverse_iteration(A, d[tgt] + 0.05, 80);
        EXPECT_NEAR(p.value, d[tgt], 1e-8) << "trial " << t;
        EXPECT_LT(residual(A, p.value, p.vector), 1e-8);
    }
}

// ---------------- Rayleigh quotient iteration ----------------

TEST(RayleighQuotientIteration, ConvergesToAnEigenpair) {
    std::mt19937 rng(3);
    for (int t = 0; t < 300; ++t) {
        const int n = 3 + static_cast<int>(rng() % 4);
        std::vector<double> d(n);
        std::uniform_real_distribution<double> u(-10, 10);
        for (auto& x : d) x = u(rng);
        Mat Q;
        const Mat A = build_symmetric(rng, d, Q);
        std::vector<double> x0(n);
        for (int i = 0; i < n; ++i) x0[i] = u(rng);
        const auto p = rayleigh_quotient_iteration(A, x0, 40);
        // converges to some eigenpair: value is in the spectrum, residual ~0
        double best = 1e9;
        for (double dv : d) best = std::min(best, std::fabs(p.value - dv));
        EXPECT_LT(best, 1e-7) << "trial " << t;
        EXPECT_LT(residual(A, p.value, p.vector), 1e-7);
    }
}

TEST(RayleighQuotientIteration, IsCubicallyConvergent) {
    std::mt19937 rng(4);
    Mat          Q;
    const Mat    A = build_symmetric(rng, {1.0, 3.0, 7.0}, Q);
    std::vector<double> res;
    rayleigh_quotient_iteration(A, {0.4, 0.5, 0.9}, 8, &res);
    ASSERT_GE(res.size(), 5u);
    // Once in the convergence regime, each residual is far smaller than the last.
    EXPECT_LT(res[3], 1e-3);
    EXPECT_LT(res[4], 1e-9); // roughly cubed
}
