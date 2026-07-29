#include <gtest/gtest.h>

#include <datamunge/algorithms/davis_putnam.hpp>
#include <datamunge/algorithms/dpll.hpp>
#include <datamunge/algorithms/gauss_newton.hpp>
#include <datamunge/algorithms/grasp.hpp>

#include <cmath>
#include <random>
#include <tuple>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Gauss-Newton ----------------

TEST(GaussNewton, FitsExponentialModel) {
    auto model = [](const std::vector<double>& p, double x) { return p[0] * std::exp(p[1] * x); };
    std::vector<double> xs, ys;
    for (int i = 0; i < 20; ++i) { const double x = i * 0.2; xs.push_back(x); ys.push_back(2.5 * std::exp(0.4 * x)); }
    const auto r = gauss_newton(model, xs, ys, {1.0, 0.1}, 100, 1e-14);
    EXPECT_NEAR(r.params[0], 2.5, 1e-6);
    EXPECT_NEAR(r.params[1], 0.4, 1e-6);
    EXPECT_LT(r.sse, 1e-18);
}

TEST(GaussNewton, FitsSinusoidalModel) {
    auto model = [](const std::vector<double>& p, double x) { return p[0] * std::sin(p[1] * x + p[2]); };
    std::vector<double> xs, ys;
    for (int i = 0; i < 40; ++i) { const double x = i * 0.15; xs.push_back(x); ys.push_back(3.0 * std::sin(1.2 * x + 0.5)); }
    const auto r = gauss_newton(model, xs, ys, {2.5, 1.0, 0.3}, 200, 1e-14);
    EXPECT_NEAR(r.params[0], 3.0, 1e-5);
    EXPECT_NEAR(r.params[1], 1.2, 1e-5);
    EXPECT_NEAR(r.params[2], 0.5, 1e-5);
}

// ---------------- Davis-Putnam ----------------

TEST(DavisPutnam, BasicCases) {
    EXPECT_TRUE(davis_putnam(3, {{1, 2}, {-1, 3}}));
    EXPECT_FALSE(davis_putnam(1, {{1}, {-1}}));
    EXPECT_FALSE(davis_putnam(2, {{1, 2}, {1, -2}, {-1, 2}, {-1, -2}}));
    EXPECT_TRUE(davis_putnam(3, {})); // no clauses -> trivially satisfiable
}

TEST(DavisPutnam, AgreesWithDpllOnSmallRandom3Sat) {
    std::mt19937 rng(1);
    for (int t = 0; t < 500; ++t) {
        const int nv = 3 + static_cast<int>(rng() % 3); // 3..5 vars (DP resolution stays tractable)
        const int nc = nv * 3 + static_cast<int>(rng() % 3);
        std::vector<std::vector<int>> cls;
        for (int c = 0; c < nc; ++c) {
            std::vector<int> cl;
            for (int k = 0; k < 3; ++k) { const int v = 1 + static_cast<int>(rng() % nv); cl.push_back((rng() & 1) ? v : -v); }
            cls.push_back(cl);
        }
        EXPECT_EQ(davis_putnam(nv, cls), dpll_sat(nv, cls).satisfiable) << "trial " << t;
    }
}

// ---------------- GRASP (max-cut) ----------------

TEST(Grasp, ReachesBruteForceOptimumOnSmallGraphs) {
    std::mt19937 rng(1);
    int          reached = 0; const int total = 60;
    for (int t = 0; t < total; ++t) {
        const int n = 6 + static_cast<int>(rng() % 6); // 6..11 vertices
        std::vector<std::vector<std::pair<int, double>>> adj(n);
        std::vector<std::tuple<int, int, double>>        edges;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (rng() % 2) {
                    const double w = 1 + static_cast<double>(rng() % 9);
                    adj[i].push_back({j, w});
                    adj[j].push_back({i, w});
                    edges.emplace_back(i, j, w);
                }
        double opt = 0;
        for (unsigned m = 0; m < (1u << n); ++m) {
            double c = 0;
            for (auto& [a, b, w] : edges) if (((m >> a) & 1) != ((m >> b) & 1)) c += w;
            opt = std::max(opt, c);
        }
        const auto r = grasp_max_cut(n, adj, 80, 0.3, 1 + t);
        // valid cut value
        double check = 0;
        for (auto& [a, b, w] : edges) if (r.side[a] != r.side[b]) check += w;
        EXPECT_NEAR(check, r.value, 1e-9);
        if (r.value >= opt - 1e-9) ++reached;
    }
    EXPECT_GE(reached, static_cast<int>(0.9 * total)); // near-optimal on the vast majority
}
