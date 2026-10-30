#include <gtest/gtest.h>

#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/hull_algorithms.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::geometry;

namespace {

// Canonicalise a CCW hull: rotate so it starts at the lexicographically smallest
// vertex, so two hulls of the same shape compare equal regardless of start.
std::vector<Point2D> canon(std::vector<Point2D> h) {
    if (h.size() < 2) return h;
    std::size_t s = 0;
    for (std::size_t i = 1; i < h.size(); ++i)
        if (h[i].x < h[s].x || (h[i].x == h[s].x && h[i].y < h[s].y)) s = i;
    std::rotate(h.begin(), h.begin() + s, h.end());
    return h;
}

bool eq(const std::vector<Point2D>& a, const std::vector<Point2D>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

} // namespace

TEST(HullAlgorithms, AgreeWithMonotoneChainRandom) {
    std::mt19937_64                    rng(1);
    std::uniform_int_distribution<int> coord(-50, 50);
    for (int t = 0; t < 500; ++t) {
        const int            npts = 3 + static_cast<int>(rng() % 60);
        std::vector<Point2D> pts;
        for (int i = 0; i < npts; ++i) pts.push_back({static_cast<double>(coord(rng)), static_cast<double>(coord(rng))});

        const auto ref = canon(convex_hull(pts));
        EXPECT_TRUE(eq(canon(graham_scan(pts)), ref)) << "graham t=" << t;
        EXPECT_TRUE(eq(canon(jarvis_march(pts)), ref)) << "jarvis t=" << t;
        EXPECT_TRUE(eq(canon(chans_algorithm(pts)), ref)) << "chan t=" << t;
    }
}

TEST(HullAlgorithms, LargerAndDenser) {
    std::mt19937_64                    rng(2);
    std::uniform_int_distribution<int> coord(-1000, 1000);
    for (int t = 0; t < 40; ++t) {
        std::vector<Point2D> pts;
        for (int i = 0; i < 400; ++i) pts.push_back({static_cast<double>(coord(rng)), static_cast<double>(coord(rng))});
        const auto ref = canon(convex_hull(pts));
        EXPECT_TRUE(eq(canon(graham_scan(pts)), ref)) << "graham t=" << t;
        EXPECT_TRUE(eq(canon(jarvis_march(pts)), ref)) << "jarvis t=" << t;
        EXPECT_TRUE(eq(canon(chans_algorithm(pts)), ref)) << "chan t=" << t;
    }
}

TEST(HullAlgorithms, CollinearAndDuplicatesExcluded) {
    // Points on a line plus duplicates: hull is the two extreme endpoints.
    std::vector<Point2D> line = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {2, 2}, {1, 1}};
    const auto           ref  = canon(convex_hull(line));
    EXPECT_TRUE(eq(canon(graham_scan(line)), ref));
    EXPECT_TRUE(eq(canon(jarvis_march(line)), ref));
    EXPECT_TRUE(eq(canon(chans_algorithm(line)), ref));

    // Square with a point on an edge and an interior point: hull is the 4 corners.
    std::vector<Point2D> sq = {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 0}, {2, 2}};
    const auto           r2 = canon(convex_hull(sq));
    EXPECT_EQ(r2.size(), 4u);
    EXPECT_TRUE(eq(canon(graham_scan(sq)), r2));
    EXPECT_TRUE(eq(canon(jarvis_march(sq)), r2));
    EXPECT_TRUE(eq(canon(chans_algorithm(sq)), r2));
}

TEST(HullAlgorithms, KnownTriangleAndSquare) {
    std::vector<Point2D> tri = {{0, 0}, {2, 0}, {1, 2}, {1, 1}}; // interior (1,1) dropped
    EXPECT_EQ(graham_scan(tri).size(), 3u);
    EXPECT_EQ(jarvis_march(tri).size(), 3u);
    EXPECT_EQ(chans_algorithm(tri).size(), 3u);
}

TEST(HullAlgorithms, TangentMatchesBruteForce) {
    std::mt19937_64                    rng(7);
    std::uniform_int_distribution<int> coord(-30, 30);
    int                                checked = 0;
    for (int t = 0; t < 4000; ++t) {
        std::vector<Point2D> pts;
        const int            npts = 3 + static_cast<int>(rng() % 80);
        for (int i = 0; i < npts; ++i) pts.push_back({double(coord(rng)), double(coord(rng))});
        auto poly = convex_hull(pts);
        if (poly.size() < 3) continue;
        std::rotate(poly.begin(), poly.begin() + rng() % poly.size(), poly.end()); // any start
        const Point2D p{double(coord(rng) * 2), double(coord(rng) * 2)};
        // p must be strictly outside: some edge has p strictly on its right.
        bool outside = false;
        for (std::size_t i = 0; i < poly.size(); ++i)
            if (cross(poly[i], poly[(i + 1) % poly.size()], p) < 0) outside = true;
        if (!outside) continue;
        // Brute force: the vertex with every other vertex left of or on p->v, farthest on ties.
        int best = -1;
        for (int i = 0; i < int(poly.size()); ++i) {
            bool ok = true;
            for (const auto& q : poly)
                if (cross(p, poly[i], q) < 0) ok = false;
            if (ok && (best < 0 || squared_distance(p, poly[i]) > squared_distance(p, poly[best]))) best = i;
        }
        const auto ts = convex_polygon_tangent(poly, p);
        ASSERT_EQ(ts.index, best) << "t=" << t << " k=" << poly.size();
        EXPECT_LE(ts.orientation_tests, 6 * (std::bit_width(poly.size()) + 3)) << "t=" << t;
        ++checked;
    }
    EXPECT_GT(checked, 2000);
}

TEST(HullAlgorithms, ChanTraceRounds) {
    for (const auto cloud : {PointCloud::Square, PointCloud::Disk, PointCloud::Gaussian, PointCloud::Parabola}) {
        const auto pts = random_point_cloud(20000, cloud, 11);
        const auto ref = canon(convex_hull(pts));
        const auto res = chans_algorithm_trace(pts);
        EXPECT_TRUE(eq(canon(res.hull), ref));
        ChanOptions scan;
        scan.binary_tangent = false;
        const auto lin = chans_algorithm_trace(pts, scan);
        EXPECT_TRUE(eq(lin.hull, res.hull));
        ASSERT_FALSE(res.rounds.empty());
        long long m = 4;
        for (std::size_t r = 0; r < res.rounds.size(); ++r) {
            EXPECT_EQ(res.rounds[r].m, m);
            EXPECT_EQ(res.rounds[r].completed, r + 1 == res.rounds.size());
            m = std::min<long long>(res.distinct_points, m * m);
        }
        EXPECT_EQ(res.rounds.back().wrap_steps, int(ref.size()));
        // O(n log h): the total stays within a small constant of n log2 h.
        const double nlogh = double(res.distinct_points) * std::log2(double(ref.size()));
        EXPECT_LT(res.orientation_tests, 12 * nlogh) << int(cloud);
    }
    EXPECT_EQ(int(convex_hull(random_point_cloud(5000, PointCloud::Parabola, 3)).size()), 5000);
}

TEST(HullAlgorithms, CountsAndLayers) {
    const auto pts = random_point_cloud(3000, PointCloud::Square, 5);
    long long  jt  = 0;
    const auto h   = jarvis_march(pts, jt);
    EXPECT_EQ(jt, (3000LL - 2) * static_cast<long long>(h.size()));
    long long gt = 0;
    EXPECT_TRUE(eq(graham_scan(pts, gt), graham_scan(pts)));
    EXPECT_GT(gt, 3000);

    // Nested squares plus a centre point, with a duplicate.
    std::vector<Point2D> sq = {{0, 0}, {6, 0}, {6, 6}, {0, 6}, {1, 1}, {5, 1}, {5, 5}, {1, 5}, {3, 3}, {6, 6}, {3, 0}};
    const auto           layers = convex_layers(sq);
    const std::vector<int> want = {0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1};
    EXPECT_EQ(layers, want);
}
