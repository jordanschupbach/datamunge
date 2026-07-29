#include <gtest/gtest.h>

#include <datamunge/algorithms/game_search.hpp>
#include <datamunge/algorithms/nesting.hpp>
#include <datamunge/algorithms/sss_star.hpp>
#include <datamunge/geometry/cone_algorithm.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;
using namespace datamunge::geometry;

// ---------------- SSS* ----------------

namespace {
GameNode build_tree(std::mt19937& rng, int depth) {
    GameNode n;
    if (depth == 0) { n.leaf = true; n.value = static_cast<double>(rng() % 21) - 10; return n; }
    const int k = 2 + static_cast<int>(rng() % 3);
    for (int i = 0; i < k; ++i) n.children.push_back(build_tree(rng, depth - 1));
    return n;
}
} // namespace

TEST(SssStar, MatchesMinimaxValue) {
    std::mt19937 rng(1);
    long long    mm_total = 0, sss_total = 0;
    for (int t = 0; t < 300; ++t) {
        const GameNode root = build_tree(rng, 5);
        const auto     mm   = minimax(root, true);
        const auto     ss   = sss_star(root, true, 1.0);
        EXPECT_EQ(ss.value, mm.value) << "trial " << t;
        mm_total += mm.leaves_evaluated;
        sss_total += ss.leaves_evaluated;
    }
    // SSS* dominates alpha-beta / minimax: it never examines more leaves.
    EXPECT_LT(sss_total, mm_total);
}

TEST(SssStar, MatchesAlphaBetaOnDeepTree) {
    std::mt19937 rng(42);
    for (int t = 0; t < 50; ++t) {
        const GameNode root = build_tree(rng, 6);
        EXPECT_EQ(sss_star(root, true, 1.0).value, alpha_beta(root, true).value) << "trial " << t;
    }
}

// ---------------- Nesting (First-Fit-Decreasing) ----------------

TEST(Nesting, ValidPackingAndLowerBound) {
    std::mt19937_64                        rng(3);
    std::uniform_real_distribution<double> sz(0.05, 1.0);
    const double                           cap = 1.0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> sizes;
        for (int i = 0; i < 40; ++i) sizes.push_back(sz(rng));
        const auto r = nest_first_fit_decreasing(sizes, cap);

        // every item placed exactly once
        std::vector<int> seen(sizes.size(), 0);
        int              count = 0;
        for (const auto& bin : r.assignment)
            for (int it : bin) { ++seen[it]; ++count; }
        EXPECT_EQ(count, static_cast<int>(sizes.size()));
        for (int s : seen) EXPECT_EQ(s, 1);
        // no bin exceeds capacity
        for (double load : r.loads) EXPECT_LE(load, cap + 1e-9);
        // bins within [lower bound, n]
        EXPECT_GE(r.bins, nesting_lower_bound(sizes, cap));
        EXPECT_LE(r.bins, static_cast<int>(sizes.size()));
    }
}

TEST(Nesting, OptimalOnPerfectPairs) {
    // items that pair exactly into capacity 10 -> optimum is n/2 bins
    std::vector<double> sizes{6, 4, 6, 4, 7, 3, 5, 5};
    const auto          r = nest_first_fit_decreasing(sizes, 10.0);
    EXPECT_EQ(r.bins, 4);
    EXPECT_EQ(r.bins, nesting_lower_bound(sizes, 10.0));
}

// ---------------- Cone algorithm ----------------

TEST(ConeAlgorithm, FlagsBoundaryNotInterior) {
    const double         R = 10.0;
    std::vector<Point2D> pts;
    std::mt19937_64      rng(1);
    std::uniform_real_distribution<double> u(-R, R);
    for (int i = 0; i < 3000; ++i) {
        const double x = u(rng), y = u(rng);
        if (x * x + y * y <= R * R) pts.push_back({x, y});
    }
    const auto surf = cone_surface_points(pts, 2.5, M_PI * 0.55);

    int interior_flagged = 0, extreme_total = 0, extreme_flagged = 0;
    double min_flagged_d = 1e9;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const double d = std::hypot(pts[i].x, pts[i].y);
        if (d < R - 3.0 && surf[i]) ++interior_flagged;
        if (surf[i]) min_flagged_d = std::min(min_flagged_d, d);
        if (d > R - 0.6) { ++extreme_total; if (surf[i]) ++extreme_flagged; }
    }
    EXPECT_EQ(interior_flagged, 0);                              // no interior false positives
    EXPECT_GT(min_flagged_d, R - 2.0);                          // all flagged points near the rim
    EXPECT_GT(extreme_flagged, extreme_total / 2);             // most extreme-rim points flagged
}
