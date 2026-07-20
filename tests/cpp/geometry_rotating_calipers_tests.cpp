#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

using datamunge::geometry::minimum_bounding_rectangle;
using datamunge::geometry::Point2D;
using datamunge::geometry::polygon_diameter;

// ---- polygon_diameter ----

TEST(PolygonDiameter, SquareDiagonalIsTheDiameter) {
    const std::vector<Point2D> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const auto result = polygon_diameter(square);
    EXPECT_NEAR(result.distance, std::sqrt(32.0), 1e-9);
}

TEST(PolygonDiameter, IgnoresInteriorPoints) {
    // The interior point can never be part of the diameter pair.
    const std::vector<Point2D> points{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}};
    const auto result = polygon_diameter(points);
    EXPECT_NEAR(result.distance, std::sqrt(32.0), 1e-9);
    EXPECT_NE(result.a, (Point2D{2, 2}));
    EXPECT_NE(result.b, (Point2D{2, 2}));
}

TEST(PolygonDiameter, TwoPointsAreTriviallyTheDiameter) {
    const auto result = polygon_diameter({{0, 0}, {3, 4}});
    EXPECT_DOUBLE_EQ(result.distance, 5.0);
}

TEST(PolygonDiameter, RejectsFewerThanTwoDistinctPoints) {
    EXPECT_THROW(static_cast<void>(polygon_diameter({{0, 0}})), std::invalid_argument);
}

// ---- minimum_bounding_rectangle ----

TEST(MinimumBoundingRectangle, AxisAlignedSquareGivesItsOwnArea) {
    const std::vector<Point2D> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const auto mbr = minimum_bounding_rectangle(square);
    EXPECT_NEAR(mbr.area, 16.0, 1e-9);
}

TEST(MinimumBoundingRectangle, RotatedSquareBeatsItsAxisAlignedBoundingBox) {
    // A square rotated 45 degrees ("diamond"): side length sqrt(8), so its own area is 8 --
    // the minimum bounding rectangle must find this (aligned with the diamond's own edges),
    // not the much larger axis-aligned bounding box (a 4x4 square, area 16).
    const std::vector<Point2D> diamond{{2, 0}, {4, 2}, {2, 4}, {0, 2}};
    const auto mbr = minimum_bounding_rectangle(diamond);
    EXPECT_NEAR(mbr.area, 8.0, 1e-9);
    EXPECT_LT(mbr.area, 16.0);
}

TEST(MinimumBoundingRectangle, CornersEncloseEveryInputPoint) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> unif(0.0, 10.0);
    std::vector<Point2D> points;
    for (int i = 0; i < 30; ++i) points.push_back({unif(rng), unif(rng)});

    const auto mbr = minimum_bounding_rectangle(points);
    ASSERT_EQ(mbr.corners.size(), 4u);

    // Every point must lie within the rectangle's local (u, v) frame, i.e. within
    // [0, width] x [0, height] measured from corners[0] along the two edge directions.
    const Point2D origin = mbr.corners[0];
    const double edge_len = std::sqrt((mbr.corners[1].x - origin.x) * (mbr.corners[1].x - origin.x) +
                                       (mbr.corners[1].y - origin.y) * (mbr.corners[1].y - origin.y));
    ASSERT_GT(edge_len, 0.0);
    const double dx = (mbr.corners[1].x - origin.x) / edge_len, dy = (mbr.corners[1].y - origin.y) / edge_len;
    const double perp_x = -dy, perp_y = dx;

    for (const auto& p : points) {
        const double u = (p.x - origin.x) * dx + (p.y - origin.y) * dy;
        const double v = (p.x - origin.x) * perp_x + (p.y - origin.y) * perp_y;
        EXPECT_GE(u, -1e-6);
        EXPECT_LE(u, mbr.width + 1e-6);
        EXPECT_GE(v, -1e-6);
        EXPECT_LE(v, mbr.height + 1e-6);
    }
}

TEST(MinimumBoundingRectangle, RejectsFewerThanThreeNonCollinearPoints) {
    EXPECT_THROW(static_cast<void>(minimum_bounding_rectangle({{0, 0}, {1, 1}})), std::invalid_argument);
}
