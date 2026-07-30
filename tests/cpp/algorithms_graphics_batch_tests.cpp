#include <gtest/gtest.h>

#include <datamunge/algorithms/cohen_sutherland.hpp>
#include <datamunge/algorithms/dda_line.hpp>
#include <datamunge/algorithms/midpoint_circle.hpp>
#include <datamunge/algorithms/ramer_douglas_peucker.hpp>
#include <datamunge/algorithms/sutherland_hodgman.hpp>
#include <datamunge/algorithms/xiaolin_wu_line.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

using namespace datamunge::algorithms;

TEST(DDA, RasterizesEndpointsAndDiagonal) {
    auto p = dda_line(0, 0, 10, 5);
    EXPECT_EQ(p.front(), (std::pair<int, int>{0, 0}));
    EXPECT_EQ(p.back(), (std::pair<int, int>{10, 5}));
    EXPECT_EQ(p.size(), 11u);  // steps = max(10,5) = 10 -> 11 points
    auto d = dda_line(0, 0, 4, 4);
    EXPECT_EQ(d[2], (std::pair<int, int>{2, 2}));
}

TEST(MidpointCircle, PointsLieOnTheCircle) {
    const int r = 10;
    auto      pts = midpoint_circle(0, 0, r);
    ASSERT_FALSE(pts.empty());
    for (const auto& [x, y] : pts) {
        const int d2 = x * x + y * y;
        // Every plotted cell is within a pixel of radius r: |sqrt(d2) - r| < 1.
        EXPECT_LT(std::fabs(std::sqrt(static_cast<double>(d2)) - r), 1.0);
    }
    // Cardinal points are present.
    auto has = [&](int x, int y) {
        for (const auto& p : pts) if (p.first == x && p.second == y) return true;
        return false;
    };
    EXPECT_TRUE(has(r, 0));
    EXPECT_TRUE(has(0, r));
    EXPECT_TRUE(has(-r, 0));
    EXPECT_TRUE(has(0, -r));
}

TEST(CohenSutherland, ClipsToRectangle) {
    // Rectangle [0,10]x[0,10].
    auto c = cohen_sutherland_clip(-5, 5, 15, 5, 0, 0, 10, 10);  // horizontal through the middle
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.x0, 0.0, 1e-9);
    EXPECT_NEAR(c.x1, 10.0, 1e-9);
    EXPECT_NEAR(c.y0, 5.0, 1e-9);
    // Fully inside -> unchanged.
    auto inside = cohen_sutherland_clip(2, 2, 8, 8, 0, 0, 10, 10);
    EXPECT_TRUE(inside.visible);
    EXPECT_NEAR(inside.x0, 2.0, 1e-9);
    // Fully outside (above) -> rejected.
    auto outside = cohen_sutherland_clip(-5, 20, 15, 20, 0, 0, 10, 10);
    EXPECT_FALSE(outside.visible);
}

TEST(SutherlandHodgman, ClipsPolygonToRectangle) {
    // A big triangle clipped to the unit-ish square [0,10]x[0,10].
    std::vector<Point2> tri = {{-5, -5}, {15, 5}, {5, 15}};
    auto                out = sutherland_hodgman_clip(tri, 0, 0, 10, 10);
    ASSERT_FALSE(out.empty());
    // Every output vertex lies within the clip rectangle (with tiny tolerance).
    for (const auto& [x, y] : out) {
        EXPECT_GE(x, -1e-9);
        EXPECT_LE(x, 10 + 1e-9);
        EXPECT_GE(y, -1e-9);
        EXPECT_LE(y, 10 + 1e-9);
    }
    // A polygon already inside is returned unchanged in extent.
    std::vector<Point2> sq = {{2, 2}, {8, 2}, {8, 8}, {2, 8}};
    auto                same = sutherland_hodgman_clip(sq, 0, 0, 10, 10);
    EXPECT_EQ(same.size(), 4u);
}

TEST(RamerDouglasPeucker, DropsCollinearKeepsCorners) {
    // A polyline that is a straight run then a corner; RDP should keep only the corners.
    std::vector<RDPPoint> pts = {{0, 0}, {1, 0.01}, {2, 0}, {3, 0}, {4, 0}, {5, 2}, {6, 4}};
    auto                  simp = ramer_douglas_peucker(pts, 0.1);
    // The nearly-collinear run collapses; endpoints and the {4,0} corner survive.
    EXPECT_LT(simp.size(), pts.size());
    EXPECT_EQ(simp.front(), (RDPPoint{0, 0}));
    EXPECT_EQ(simp.back(), (RDPPoint{6, 4}));
    // A tiny epsilon keeps (almost) everything; a huge epsilon keeps only the endpoints.
    EXPECT_EQ(ramer_douglas_peucker(pts, 1000.0).size(), 2u);
}

TEST(XiaolinWu, CoverageSumsToOnePerStep) {
    auto px = xiaolin_wu_line(0, 0, 10, 4);
    ASSERT_FALSE(px.empty());
    // Pixels come in pairs (near/far) per x-step; each pair's brightness sums to 1.
    for (std::size_t i = 0; i + 1 < px.size(); i += 2) {
        EXPECT_NEAR(px[i].brightness + px[i + 1].brightness, 1.0, 1e-9);
        EXPECT_GE(px[i].brightness, -1e-9);
        EXPECT_LE(px[i].brightness, 1.0 + 1e-9);
    }
}
