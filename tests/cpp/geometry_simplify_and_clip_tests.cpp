#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

using datamunge::geometry::clip_polygon;
using datamunge::geometry::Point2D;
using datamunge::geometry::polygon_area;
using datamunge::geometry::simplify_polyline;

// ---- simplify_polyline (Douglas-Peucker) ----

TEST(SimplifyPolyline, RemovesPointsCloseToTheBaselineKeepsFarOutliers) {
    // A single spike at (2,5): the two near-baseline points on either side of it (distance
    // ~0.891 from their local sub-baseline, hand-computed) should be dropped at epsilon=1.
    const std::vector<Point2D> line{{0, 0}, {1, 0.1}, {2, 5}, {3, 0.1}, {4, 0}};
    const auto simplified = simplify_polyline(line, 1.0);
    EXPECT_EQ(simplified, (std::vector<Point2D>{{0, 0}, {2, 5}, {4, 0}}));
}

TEST(SimplifyPolyline, LargeEpsilonCollapsesToJustTheEndpoints) {
    const std::vector<Point2D> line{{0, 0}, {1, 0.1}, {2, 5}, {3, 0.1}, {4, 0}};
    const auto simplified = simplify_polyline(line, 100.0);
    EXPECT_EQ(simplified, (std::vector<Point2D>{{0, 0}, {4, 0}}));
}

TEST(SimplifyPolyline, ZeroEpsilonKeepsAllStrictlyOffLinePoints) {
    const std::vector<Point2D> line{{0, 0}, {1, 0.1}, {2, 5}, {3, 0.1}, {4, 0}};
    const auto simplified = simplify_polyline(line, 0.0);
    EXPECT_EQ(simplified.size(), line.size());
}

TEST(SimplifyPolyline, FewerThanThreePointsIsReturnedUnchanged) {
    const std::vector<Point2D> line{{0, 0}, {1, 1}};
    EXPECT_EQ(simplify_polyline(line, 0.5), line);
}

// ---- clip_polygon (Sutherland-Hodgman) ----

TEST(ClipPolygon, FullyContainedClipPolygonIsReturnedUnchanged) {
    const std::vector<Point2D> subject{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const std::vector<Point2D> clip{{1, 1}, {3, 1}, {3, 3}, {1, 3}};
    const auto result = clip_polygon(subject, clip);
    EXPECT_DOUBLE_EQ(polygon_area(result), 4.0);
}

TEST(ClipPolygon, PartiallyOverlappingSquaresIntersectToTheirCommonQuadrant) {
    const std::vector<Point2D> subject{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const std::vector<Point2D> clip{{2, 2}, {6, 2}, {6, 6}, {2, 6}};
    const auto result = clip_polygon(subject, clip);
    EXPECT_DOUBLE_EQ(polygon_area(result), 4.0); // the [2,4]x[2,4] square
}

TEST(ClipPolygon, DisjointPolygonsProduceAnEmptyResult) {
    const std::vector<Point2D> subject{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const std::vector<Point2D> clip{{10, 10}, {12, 10}, {12, 12}, {10, 12}};
    EXPECT_TRUE(clip_polygon(subject, clip).empty());
}

TEST(ClipPolygon, ClippingAgainstItselfIsAnIdentity) {
    const std::vector<Point2D> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const auto result = clip_polygon(square, square);
    EXPECT_DOUBLE_EQ(polygon_area(result), polygon_area(square));
}
