#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

using datamunge::geometry::convex_hull;
using datamunge::geometry::Point2D;

TEST(ConvexHull, SquareWithInteriorPointExcludesTheInteriorPoint) {
    const std::vector<Point2D> points{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}};
    const auto hull = convex_hull(points);

    ASSERT_EQ(hull.size(), 4u);
    EXPECT_EQ(hull, (std::vector<Point2D>{{0, 0}, {4, 0}, {4, 4}, {0, 4}})); // counterclockwise, starting from lowest-x/lowest-y
}

TEST(ConvexHull, ExcludesPointsOnAnEdge) {
    // (2,0) lies exactly on the edge between (0,0) and (4,0) -- the "strict" hull excludes it.
    const std::vector<Point2D> points{{0, 0}, {2, 0}, {4, 0}, {4, 4}, {0, 4}};
    const auto hull = convex_hull(points);

    ASSERT_EQ(hull.size(), 4u);
    for (const auto& p : hull) EXPECT_NE(p, (Point2D{2, 0}));
}

TEST(ConvexHull, TriangleIsItsOwnHull) {
    const std::vector<Point2D> points{{0, 0}, {4, 0}, {2, 4}};
    const auto hull = convex_hull(points);
    EXPECT_EQ(hull, (std::vector<Point2D>{{0, 0}, {4, 0}, {2, 4}}));
}

TEST(ConvexHull, DuplicatePointsAreDeduplicated) {
    const std::vector<Point2D> points{{0, 0}, {0, 0}, {4, 0}, {2, 4}};
    const auto hull = convex_hull(points);
    EXPECT_EQ(hull.size(), 3u);
}

TEST(ConvexHull, FewerThanThreePointsIsReturnedUnchanged) {
    const std::vector<Point2D> points{{0, 0}, {1, 1}};
    const auto hull = convex_hull(points);
    ASSERT_EQ(hull.size(), 2u);
}
