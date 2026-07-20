#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

using datamunge::geometry::point_in_polygon;
using datamunge::geometry::Point2D;
using datamunge::geometry::polygon_area;
using datamunge::geometry::segments_intersect;
using datamunge::geometry::signed_polygon_area;

// ---- point_in_polygon / polygon_area ----

TEST(Polygon, PointInPolygonForASquare) {
    const std::vector<Point2D> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    EXPECT_TRUE(point_in_polygon({2, 2}, square));
    EXPECT_FALSE(point_in_polygon({5, 5}, square));
    EXPECT_FALSE(point_in_polygon({-1, 2}, square));
}

TEST(Polygon, PointInPolygonForAConcaveLShape) {
    // An L-shape: a 4x4 square with the top-right 2x2 quadrant removed.
    const std::vector<Point2D> l_shape{{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}};
    EXPECT_TRUE(point_in_polygon({1, 1}, l_shape));   // inside the remaining L
    EXPECT_FALSE(point_in_polygon({3, 3}, l_shape));  // inside the removed notch
    EXPECT_TRUE(point_in_polygon({3, 1}, l_shape));   // inside the bottom arm of the L
}

TEST(Polygon, SignedAreaReflectsWindingOrder) {
    const std::vector<Point2D> ccw_square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const std::vector<Point2D> cw_square{{0, 0}, {0, 4}, {4, 4}, {4, 0}};

    EXPECT_DOUBLE_EQ(signed_polygon_area(ccw_square), 16.0);
    EXPECT_DOUBLE_EQ(signed_polygon_area(cw_square), -16.0);
    EXPECT_DOUBLE_EQ(polygon_area(ccw_square), 16.0);
    EXPECT_DOUBLE_EQ(polygon_area(cw_square), 16.0);
}

TEST(Polygon, AreaOfATriangle) {
    const std::vector<Point2D> triangle{{0, 0}, {4, 0}, {0, 3}};
    EXPECT_DOUBLE_EQ(polygon_area(triangle), 6.0); // 0.5 * base * height = 0.5 * 4 * 3
}

// ---- segments_intersect ----

TEST(SegmentIntersection, CrossingSegmentsIntersect) {
    EXPECT_TRUE(segments_intersect({0, 0}, {4, 4}, {0, 4}, {4, 0}));
}

TEST(SegmentIntersection, ParallelNonIntersectingSegmentsDoNotIntersect) {
    EXPECT_FALSE(segments_intersect({0, 0}, {1, 0}, {0, 1}, {1, 1}));
}

TEST(SegmentIntersection, TouchingAtAnEndpointCountsAsIntersecting) {
    EXPECT_TRUE(segments_intersect({0, 0}, {2, 2}, {2, 2}, {4, 0}));
}

TEST(SegmentIntersection, OverlappingCollinearSegmentsIntersect) {
    EXPECT_TRUE(segments_intersect({0, 0}, {4, 0}, {2, 0}, {6, 0}));
}

TEST(SegmentIntersection, DisjointCollinearSegmentsDoNotIntersect) {
    EXPECT_FALSE(segments_intersect({0, 0}, {1, 0}, {2, 0}, {3, 0}));
}

TEST(SegmentIntersection, NonCrossingSegmentsDoNotIntersect) {
    EXPECT_FALSE(segments_intersect({0, 0}, {1, 1}, {3, 0}, {4, 1}));
}
