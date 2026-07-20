#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

using datamunge::geometry::Point2D;
using datamunge::geometry::polygon_area;
using datamunge::geometry::triangulate_polygon;
using datamunge::geometry::Triangle;

namespace {
double triangle_area(const Point2D& a, const Point2D& b, const Point2D& c) {
    return std::abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0;
}

double sum_of_triangle_areas(const std::vector<Point2D>& polygon, const std::vector<Triangle>& triangles) {
    double total = 0.0;
    for (const auto& t : triangles) {
        total += triangle_area(polygon[t.a], polygon[t.b], polygon[t.c]);
    }
    return total;
}
} // namespace

TEST(TriangulatePolygon, TriangleInputIsReturnedAsItself) {
    const std::vector<Point2D> triangle{{0, 0}, {4, 0}, {0, 4}};
    const auto result = triangulate_polygon(triangle);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(sum_of_triangle_areas(triangle, result), polygon_area(triangle));
}

TEST(TriangulatePolygon, ConvexSquareProducesTwoTrianglesMatchingItsArea) {
    const std::vector<Point2D> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const auto result = triangulate_polygon(square);
    EXPECT_EQ(result.size(), 2u);
    EXPECT_DOUBLE_EQ(sum_of_triangle_areas(square, result), polygon_area(square));
}

TEST(TriangulatePolygon, ConcaveLShapeMatchesTheNMinusTwoTriangleCountAndTrueArea) {
    // An L-shape: a 4x4 square with a 2x2 notch cut from the top-right, true area = 16 - 4 = 12.
    const std::vector<Point2D> lshape{{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}};
    const auto result = triangulate_polygon(lshape);
    EXPECT_EQ(result.size(), lshape.size() - 2);
    EXPECT_DOUBLE_EQ(polygon_area(lshape), 12.0);
    EXPECT_DOUBLE_EQ(sum_of_triangle_areas(lshape, result), polygon_area(lshape));
}

TEST(TriangulatePolygon, ClockwiseWindingIsHandledJustLikeCounterclockwise) {
    const std::vector<Point2D> clockwise_square{{0, 0}, {0, 4}, {4, 4}, {4, 0}};
    const auto result = triangulate_polygon(clockwise_square);
    EXPECT_EQ(result.size(), 2u);
    EXPECT_DOUBLE_EQ(sum_of_triangle_areas(clockwise_square, result), polygon_area(clockwise_square));
}

TEST(TriangulatePolygon, FewerThanThreeVerticesProducesNoTriangles) {
    const std::vector<Point2D> segment{{0, 0}, {1, 1}};
    EXPECT_TRUE(triangulate_polygon(segment).empty());
}
