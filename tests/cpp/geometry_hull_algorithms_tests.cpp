#include <gtest/gtest.h>

#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/hull_algorithms.hpp>

#include <algorithm>
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
