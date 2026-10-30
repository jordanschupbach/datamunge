#include <gtest/gtest.h>

#include <datamunge/geometry/bentley_ottmann.hpp>
#include <datamunge/stats/correlation_test.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

using namespace datamunge::geometry;

namespace {

std::vector<std::pair<int, int>> pairs_of(const std::vector<SegmentIntersection>& xs) {
    std::vector<std::pair<int, int>> p;
    for (const auto& x : xs) p.emplace_back(x.first, x.second);
    std::ranges::sort(p);
    return p;
}

}  // namespace

TEST(BentleyOttmannSweep, MatchesBruteForceOnRandomSegments) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> coord(0.0, 100.0);
    for (int trial = 0; trial < 40; ++trial) {
        std::vector<Segment> segs;
        const int n = 5 + trial * 5;
        for (int i = 0; i < n; ++i) segs.push_back({{coord(rng), coord(rng)}, {coord(rng), coord(rng)}});
        const auto sweep = bentley_ottmann_sweep(segs);
        const auto brute = segment_intersections_brute(segs);
        ASSERT_EQ(pairs_of(sweep.intersections), pairs_of(brute)) << "trial " << trial;
        EXPECT_EQ(sweep.events, 2 * segs.size() + brute.size());
        for (std::size_t k = 1; k < sweep.intersections.size(); ++k)
            EXPECT_LE(sweep.intersections[k - 1].point.x, sweep.intersections[k].point.x + 1e-9);
    }
}

TEST(BentleyOttmannSweep, ParallelCoordinateCrossingsAreDiscordantPairs) {
    std::mt19937 rng(9);
    std::normal_distribution<double> z(0.0, 1.0);
    for (double rho : {-0.9, 0.0, 0.5, 0.99}) {
        std::vector<double> x, y;
        for (int i = 0; i < 400; ++i) {
            const double a = z(rng);
            x.push_back(std::round(a * 10) / 10);  // rounded, so there are ties
            y.push_back(rho * a + std::sqrt(1 - rho * rho) * z(rng));
        }
        const auto segs = parallel_coordinate_segments(x, y);
        std::vector<double> u, v;
        for (const auto& s : segs) {
            u.push_back(s.a.y);
            v.push_back(s.b.y);
        }
        const auto tau = datamunge::stats::kendall_tau(u, v);
        EXPECT_EQ(tau.ties_x + tau.ties_y + tau.ties_xy, 0);
        const auto sweep = bentley_ottmann_sweep(segs);
        EXPECT_EQ(static_cast<long long>(sweep.intersections.size()), tau.discordant) << "rho " << rho;
    }
}

TEST(KendallTau, MatchesQuadraticCountWithTies) {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> d(0, 6);
    std::vector<double> x, y;
    for (int i = 0; i < 300; ++i) {
        x.push_back(d(rng));
        y.push_back(d(rng) + 0.5 * x.back());
    }
    long long c = 0, dis = 0, tx = 0, ty = 0, txy = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = i + 1; j < x.size(); ++j) {
            const double s = (x[i] - x[j]) * (y[i] - y[j]);
            if (x[i] == x[j] && y[i] == y[j]) ++txy;
            else if (x[i] == x[j]) ++tx;
            else if (y[i] == y[j]) ++ty;
            else if (s > 0) ++c;
            else ++dis;
        }
    const auto r = datamunge::stats::kendall_tau(x, y);
    EXPECT_EQ(r.concordant, c);
    EXPECT_EQ(r.discordant, dis);
    EXPECT_EQ(r.ties_x, tx);
    EXPECT_EQ(r.ties_y, ty);
    EXPECT_EQ(r.ties_xy, txy);
    const double n0 = 300.0 * 299.0 / 2.0;
    EXPECT_NEAR(r.tau_b, (c - dis) / std::sqrt((n0 - tx - txy) * (n0 - ty - txy)), 1e-12);
}

TEST(BentleyOttmannSweep, ThreeSegmentsThroughOnePoint) {
    // Three segments through (1, 1), and a fourth that crosses two of them further right.
    const std::vector<Segment> segs = {{{0, 0}, {2, 2}}, {{0, 2}, {2, 0}}, {{0, 1}, {2.5, 1}}, {{1.5, 3}, {3, -1}}};
    const auto sweep = bentley_ottmann_sweep(segs);
    const auto brute = segment_intersections_brute(segs);
    EXPECT_EQ(pairs_of(sweep.intersections), pairs_of(brute));
    EXPECT_EQ(brute.size(), 5u);
    EXPECT_EQ(sweep.events, 2 * segs.size() + 3);  // one event for the triple point, two for the others
}
