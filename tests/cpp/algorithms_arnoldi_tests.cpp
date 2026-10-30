#include <gtest/gtest.h>

#include <datamunge/algorithms/arnoldi_iteration.hpp>
#include <datamunge/stats/knn_graph.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using namespace datamunge::algorithms;
using datamunge::stats::KnnGraph;

namespace {

std::vector<std::vector<double>> diagonal_plus_noise(int n, unsigned seed) {
    std::mt19937                     rng(seed);
    std::normal_distribution<double> z(0.0, 0.01);
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) A[i][j] = z(rng);
        A[i][i] += 1.0 + i;  // eigenvalues near 1..n
    }
    return A;
}

}  // namespace

TEST(Arnoldi, MatrixFreeMatchesDense) {
    const auto A = diagonal_plus_noise(12, 3);
    std::vector<double> b(12, 1.0);
    MatVec mv = [&](const std::vector<double>& x) {
        std::vector<double> y(x.size(), 0.0);
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
        return y;
    };
    const auto d = arnoldi_iteration(A, b, 6);
    const auto f = arnoldi_iteration(mv, b, 6);
    ASSERT_EQ(d.H.size(), f.H.size());
    for (std::size_t i = 0; i < d.H.size(); ++i)
        for (std::size_t j = 0; j < d.H[i].size(); ++j) EXPECT_NEAR(d.H[i][j], f.H[i][j], 1e-12);
    EXPECT_LT(arnoldi_relation_residual(mv, f), 1e-12);
    EXPECT_LT(arnoldi_orthogonality_loss(f), 1e-12);
    EXPECT_THROW(arnoldi_iteration(mv, b, 0), std::invalid_argument);
}

TEST(Arnoldi, RitzResidualEstimateIsExact) {
    const auto A = diagonal_plus_noise(30, 5);
    MatVec mv = [&](const std::vector<double>& x) {
        std::vector<double> y(x.size(), 0.0);
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
        return y;
    };
    std::vector<double> b(30);
    for (int i = 0; i < 30; ++i) b[i] = 1.0 + 0.1 * i;
    const auto r     = arnoldi_iteration(mv, b, 10, true);
    const auto pairs = ritz_pairs(r);
    ASSERT_EQ(pairs.size(), 10u);
    EXPECT_GE(std::abs(pairs[0].value), std::abs(pairs[9].value));
    EXPECT_NEAR(pairs[0].value.real(), 30.0, 0.2);  // the largest eigenvalue converges first
    for (const auto& p : pairs) {
        // || A x - theta x || computed directly
        std::vector<double> xr(30), xi(30);
        for (int k = 0; k < 30; ++k) {
            xr[k] = p.vector[k].real();
            xi[k] = p.vector[k].imag();
        }
        const auto ar = mv(xr), ai = mv(xi);
        double     s = 0.0;
        for (int k = 0; k < 30; ++k) {
            const std::complex<double> res = std::complex<double>(ar[k], ai[k]) - p.value * p.vector[k];
            s += std::norm(res);
        }
        EXPECT_NEAR(std::sqrt(s), p.residual, 1e-9);
    }
}

TEST(Arnoldi, BreakdownOnInvariantSubspace) {
    // b in the span of two eigenvectors -> breakdown after 2 steps; Ritz values are exact.
    std::vector<std::vector<double>> A = {{2, 0, 0}, {0, 3, 0}, {0, 0, 5}};
    const auto r = arnoldi_iteration(A, {1, 1, 0}, 3);
    EXPECT_EQ(r.V.size(), 2u);
    const auto pairs = ritz_pairs(r, false);
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_NEAR(pairs[0].value.real(), 3.0, 1e-12);
    EXPECT_NEAR(pairs[1].value.real(), 2.0, 1e-12);
    EXPECT_NEAR(pairs[0].residual, 0.0, 1e-12);
    EXPECT_TRUE(pairs[0].vector.empty());
}

TEST(Arnoldi, BasisStaysOrthonormalOnGraphWalk) {
    // Grid-graph random walk with many converging Ritz values.
    std::vector<std::vector<double>> pts;
    for (int i = 0; i < 400; ++i) pts.push_back({static_cast<double>(i % 20) + 0.001 * i, static_cast<double>(i / 20)});
    KnnGraph g(pts, 4);
    MatVec   mv = [&](const std::vector<double>& x) { return g.random_walk_apply(x); };
    std::vector<double> b(400);
    for (int i = 0; i < 400; ++i) b[i] = std::sin(0.37 * i) + 1.0;
    const auto plain = arnoldi_iteration(mv, b, 120, false);
    const auto twice = arnoldi_iteration(mv, b, 120, true);
    EXPECT_LT(arnoldi_orthogonality_loss(twice), 1e-12);
    EXPECT_LT(arnoldi_orthogonality_loss(plain), 1e-10);
    EXPECT_LT(arnoldi_relation_residual(mv, plain), 1e-12);
}

TEST(KnnGraphTest, StructureAndRandomWalk) {
    // Two well-separated clusters of 5 points on a line.
    std::vector<std::vector<double>> pts;
    for (int i = 0; i < 5; ++i) pts.push_back({static_cast<double>(i)});
    for (int i = 0; i < 5; ++i) pts.push_back({100.0 + i});
    KnnGraph g(pts, 2);
    EXPECT_EQ(g.size(), 10u);
    EXPECT_EQ(g.components(), 2u);
    EXPECT_EQ(g.neighbors(0), (std::vector<std::size_t>{1, 2}));
    // P maps constants to constants (rows sum to one).
    const auto y = g.random_walk_apply(std::vector<double>(10, 3.0));
    for (double v : y) EXPECT_NEAR(v, 3.0, 1e-12);
    const auto spec = g.random_walk_spectrum();
    ASSERT_EQ(spec.size(), 10u);
    EXPECT_NEAR(spec[0], 1.0, 1e-10);
    EXPECT_NEAR(spec[1], 1.0, 1e-10);  // one unit eigenvalue per component
    EXPECT_LT(spec[2], 1.0 - 1e-3);
    EXPECT_GE(spec.back(), -1.0 - 1e-10);
    EXPECT_THROW(KnnGraph(pts, 10), std::invalid_argument);
}

TEST(Arnoldi, ComplexRitzValuesOfARotation) {
    // A 90-degree rotation in the plane, scaled by 2 in a third direction.
    std::vector<std::vector<double>> A = {{0, -1, 0}, {1, 0, 0}, {0, 0, 2}};
    const auto pairs = ritz_pairs(arnoldi_iteration(A, {1, 0, 1}, 3));
    ASSERT_EQ(pairs.size(), 3u);
    EXPECT_NEAR(pairs[0].value.real(), 2.0, 1e-12);
    EXPECT_NEAR(std::abs(pairs[1].value.imag()), 1.0, 1e-12);
    EXPECT_NEAR(pairs[1].value.real(), 0.0, 1e-12);
    EXPECT_NEAR((pairs[1].value * pairs[2].value).real(), 1.0, 1e-12);  // conjugate pair
    EXPECT_NEAR(pairs[1].residual, 0.0, 1e-12);
}
