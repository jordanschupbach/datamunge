#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <cmath>
#include <random>

using datamunge::geometry::delaunay_triangulation;
using datamunge::geometry::distance;
using datamunge::geometry::Point2D;
using datamunge::geometry::Triangle;

namespace {
// An independent circumcenter/circumradius computation (deliberately not reusing the
// implementation's own detail::circumcenter/in_circumcircle) used purely to VERIFY the
// defining Delaunay property -- no point strictly inside any triangle's circumcircle -- for
// whatever triangulation delaunay_triangulation() actually produces.
Point2D independent_circumcenter(const Point2D& a, const Point2D& b, const Point2D& c) {
    const double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    const double a2 = a.x * a.x + a.y * a.y;
    const double b2 = b.x * b.x + b.y * b.y;
    const double c2 = c.x * c.x + c.y * c.y;
    return Point2D{(a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
                   (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d};
}

bool satisfies_delaunay_property(const std::vector<Point2D>& points, const std::vector<Triangle>& triangles) {
    for (const auto& t : triangles) {
        const Point2D center = independent_circumcenter(points[t.a], points[t.b], points[t.c]);
        const double radius = distance(center, points[t.a]);
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (i == t.a || i == t.b || i == t.c) continue;
            if (distance(center, points[i]) < radius - 1e-9) return false; // strictly inside the circumcircle
        }
    }
    return true;
}
} // namespace

TEST(DelaunayTriangulation, SquarePlusCenterProducesFourTrianglesSatisfyingTheEmptyCircumcircleProperty) {
    const std::vector<Point2D> points{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}};
    const auto triangles = delaunay_triangulation(points);

    EXPECT_EQ(triangles.size(), 4u); // center point splits the square into exactly 4 triangles
    EXPECT_TRUE(satisfies_delaunay_property(points, triangles));
}

TEST(DelaunayTriangulation, RandomPointsSatisfyTheEmptyCircumcircleProperty) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> unif(0.0, 100.0);
    std::vector<Point2D> points;
    for (int i = 0; i < 50; ++i) points.push_back({unif(rng), unif(rng)});

    const auto triangles = delaunay_triangulation(points);
    EXPECT_TRUE(satisfies_delaunay_property(points, triangles));

    // Every triangle vertex should be a valid point index, and every triangle non-degenerate
    // (nonzero area).
    for (const auto& t : triangles) {
        ASSERT_LT(t.a, points.size());
        ASSERT_LT(t.b, points.size());
        ASSERT_LT(t.c, points.size());
        const double area2 = std::abs((points[t.b].x - points[t.a].x) * (points[t.c].y - points[t.a].y) -
                                       (points[t.c].x - points[t.a].x) * (points[t.b].y - points[t.a].y));
        EXPECT_GT(area2, 0.0);
    }
}

TEST(DelaunayTriangulation, TrianglesAreCounterclockwise) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> unif(0.0, 10.0);
    std::vector<Point2D> points;
    for (int i = 0; i < 20; ++i) points.push_back({unif(rng), unif(rng)});

    const auto triangles = delaunay_triangulation(points);
    for (const auto& t : triangles) {
        const double cross_z = (points[t.b].x - points[t.a].x) * (points[t.c].y - points[t.a].y) -
                                (points[t.b].y - points[t.a].y) * (points[t.c].x - points[t.a].x);
        EXPECT_GT(cross_z, 0.0);
    }
}

TEST(DelaunayTriangulation, FewerThanThreePointsProducesNoTriangles) {
    EXPECT_TRUE(delaunay_triangulation({{0, 0}, {1, 1}}).empty());
    EXPECT_TRUE(delaunay_triangulation({}).empty());
}
