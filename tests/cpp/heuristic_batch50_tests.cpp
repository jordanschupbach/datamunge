#include <gtest/gtest.h>

#include <datamunge/algorithms/difference_map.hpp>
#include <datamunge/algorithms/memetic.hpp>
#include <datamunge/algorithms/random_restart_hill_climbing.hpp>

#include <cmath>
#include <random>
#include <tuple>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Random-restart hill climbing ----------------

TEST(RandomRestartHillClimbing, FindsGlobalMaxOfMultimodal) {
    auto f = [](const std::vector<double>& x) { const double t = x[0]; return std::sin(t) + std::sin(2.2 * t) + 0.3 * t; };
    // brute-force global optimum on [0,15]
    double bx = 0, bf = -1e9;
    for (double t = 0; t <= 15; t += 0.0005) { const double v = std::sin(t) + std::sin(2.2 * t) + 0.3 * t; if (v > bf) { bf = v; bx = t; } }
    const auto r = random_restart_hill_climbing(f, {0.0}, {15.0}, 60, 1);
    EXPECT_NEAR(r.value, bf, 1e-3);
    EXPECT_NEAR(r.x[0], bx, 1e-2);
}

TEST(RandomRestartHillClimbing, TwoDimensionalPeak) {
    // f = -(x-3)^2 - (y+1)^2 + 5 has its max 5 at (3,-1).
    auto f = [](const std::vector<double>& x) { return -(x[0] - 3) * (x[0] - 3) - (x[1] + 1) * (x[1] + 1) + 5; };
    const auto r = random_restart_hill_climbing(f, {-10, -10}, {10, 10}, 40, 2);
    EXPECT_NEAR(r.value, 5.0, 1e-4);
    EXPECT_NEAR(r.x[0], 3.0, 1e-2);
    EXPECT_NEAR(r.x[1], -1.0, 1e-2);
}

// ---------------- Difference map ----------------

TEST(DifferenceMap, IntersectsTwoLines) {
    // A: y = 3 (project sets y),  B: x = 2 (project sets x)  ->  intersection (2,3).
    auto pA = [](const std::vector<double>& x) { return std::vector<double>{x[0], 3.0}; };
    auto pB = [](const std::vector<double>& x) { return std::vector<double>{2.0, x[1]}; };
    const auto r = difference_map(pA, pB, {-5.0, 10.0}, 1.0, 10000, 1e-13);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.solution[0], 2.0, 1e-9);
    EXPECT_NEAR(r.solution[1], 3.0, 1e-9);
}

TEST(DifferenceMap, IntersectsSlantedLines) {
    // A: y = x,  B: y = -x + 4  ->  intersection (2,2).
    auto pA = [](const std::vector<double>& p) { const double t = (p[0] + p[1]) / 2; return std::vector<double>{t, t}; };
    auto pB = [](const std::vector<double>& p) { const double t = (p[0] - p[1] + 4) / 2; return std::vector<double>{t, -t + 4}; };
    const auto r = difference_map(pA, pB, {7.0, -3.0}, 1.0, 10000, 1e-13);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.solution[0], 2.0, 1e-8);
    EXPECT_NEAR(r.solution[1], 2.0, 1e-8);
}

// ---------------- Memetic algorithm ----------------

TEST(Memetic, ReachesBruteForceMaxCut) {
    std::mt19937 rng(1);
    int          reached = 0; const int total = 40;
    for (int t = 0; t < total; ++t) {
        const int n = 6 + static_cast<int>(rng() % 6);
        std::vector<std::vector<std::pair<int, double>>> adj(n);
        std::vector<std::tuple<int, int, double>>        edges;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (rng() % 2) { const double w = 1 + static_cast<double>(rng() % 9); adj[i].push_back({j, w}); adj[j].push_back({i, w}); edges.emplace_back(i, j, w); }
        double opt = 0;
        for (unsigned m = 0; m < (1u << n); ++m) { double c = 0; for (auto& [a, b, w] : edges) if (((m >> a) & 1) != ((m >> b) & 1)) c += w; opt = std::max(opt, c); }
        const auto mr = memetic_max_cut(n, adj, 20, 30, 1 + t);
        // valid cut value
        double check = 0;
        for (auto& [a, b, w] : edges) if (mr.side[a] != mr.side[b]) check += w;
        EXPECT_NEAR(check, mr.value, 1e-9);
        if (mr.value >= opt - 1e-9) ++reached;
    }
    EXPECT_GE(reached, static_cast<int>(0.9 * total));
}
