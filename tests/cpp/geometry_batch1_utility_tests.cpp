#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <stdexcept>

using datamunge::geometry::bounding_box;
using datamunge::geometry::is_convex_polygon;
using datamunge::geometry::line_intersection_point;
using datamunge::geometry::Point2D;
using datamunge::geometry::polygon_centroid;
using datamunge::geometry::segment_intersection_point;

// ---- line_intersection_point / segment_intersection_point ----

TEST(LineIntersectionPoint, CrossingLinesIntersectAtTheExpectedPoint) {
    Point2D out;
    ASSERT_TRUE(line_intersection_point({0, 0}, {4, 4}, {0, 4}, {4, 0}, out));
    EXPECT_DOUBLE_EQ(out.x, 2.0);
    EXPECT_DOUBLE_EQ(out.y, 2.0);
}

TEST(LineIntersectionPoint, ParallelLinesHaveNoIntersection) {
    Point2D out;
    EXPECT_FALSE(line_intersection_point({0, 0}, {1, 0}, {0, 1}, {1, 1}, out));
}

TEST(LineIntersectionPoint, TreatsInputsAsInfiniteLinesUnlikeTheSegmentVersion) {
    // These "segments" don't overlap in range, but the infinite lines through them still cross.
    Point2D out;
    ASSERT_TRUE(line_intersection_point({0, 0}, {1, 1}, {3, 0}, {4, -1}, out));
}

TEST(SegmentIntersectionPoint, CrossingSegmentsIntersectAtTheExpectedPoint) {
    Point2D out;
    ASSERT_TRUE(segment_intersection_point({0, 0}, {4, 4}, {0, 4}, {4, 0}, out));
    EXPECT_DOUBLE_EQ(out.x, 2.0);
    EXPECT_DOUBLE_EQ(out.y, 2.0);
}

TEST(SegmentIntersectionPoint, NonCrossingSegmentsHaveNoIntersection) {
    Point2D out;
    EXPECT_FALSE(segment_intersection_point({0, 0}, {1, 1}, {3, 0}, {4, 1}, out));
}

TEST(SegmentIntersectionPoint, CollinearOverlapHasNoSinglePointEvenThoughSegmentsIntersect) {
    Point2D out;
    EXPECT_TRUE(datamunge::geometry::segments_intersect({0, 0}, {4, 0}, {2, 0}, {6, 0}));
    EXPECT_FALSE(segment_intersection_point({0, 0}, {4, 0}, {2, 0}, {6, 0}, out));
}

// ---- polygon_centroid ----

TEST(PolygonCentroid, SquareCentroidIsItsCenter) {
    const std::vector<Point2D> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const auto c = polygon_centroid(square);
    EXPECT_DOUBLE_EQ(c.x, 2.0);
    EXPECT_DOUBLE_EQ(c.y, 2.0);
}

TEST(PolygonCentroid, TriangleCentroidMatchesTheVertexAverage) {
    // The area centroid of a TRIANGLE specifically coincides with the plain average of its
    // vertices -- a fact special to triangles (not true for polygons in general), used here
    // as an independent, hand-computable check.
    const std::vector<Point2D> triangle{{0, 0}, {6, 0}, {0, 6}};
    const auto c = polygon_centroid(triangle);
    EXPECT_DOUBLE_EQ(c.x, 2.0);
    EXPECT_DOUBLE_EQ(c.y, 2.0);
}

TEST(PolygonCentroid, RejectsZeroAreaPolygon) {
    const std::vector<Point2D> degenerate{{0, 0}, {1, 0}, {2, 0}};
    EXPECT_THROW(static_cast<void>(polygon_centroid(degenerate)), std::invalid_argument);
}

// ---- is_convex_polygon ----

TEST(IsConvexPolygon, SquareIsConvex) {
    EXPECT_TRUE(is_convex_polygon({{0, 0}, {4, 0}, {4, 4}, {0, 4}}));
}

TEST(IsConvexPolygon, LShapeIsNotConvex) {
    EXPECT_FALSE(is_convex_polygon({{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}}));
}

TEST(IsConvexPolygon, FewerThanThreeVerticesIsNeverConvex) {
    EXPECT_FALSE(is_convex_polygon({{0, 0}, {1, 1}}));
    EXPECT_FALSE(is_convex_polygon({}));
}

// ---- bounding_box ----

TEST(BoundingBoxTest, ComputesMinAndMaxCoordinates) {
    const std::vector<Point2D> points{{3, -2}, {-1, 5}, {4, 4}, {0, 0}};
    const auto bb = bounding_box(points);
    EXPECT_DOUBLE_EQ(bb.min.x, -1.0);
    EXPECT_DOUBLE_EQ(bb.min.y, -2.0);
    EXPECT_DOUBLE_EQ(bb.max.x, 4.0);
    EXPECT_DOUBLE_EQ(bb.max.y, 5.0);
}

TEST(BoundingBoxTest, RejectsEmptyPointSet) {
    EXPECT_THROW(static_cast<void>(bounding_box({})), std::invalid_argument);
}
