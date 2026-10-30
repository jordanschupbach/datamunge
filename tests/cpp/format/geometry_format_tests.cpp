// std::format / std::print support for datamunge::geometry.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <vector>

namespace geo = dm::geometry;

static_assert(std::formattable<geo::Point2D, char>);
static_assert(std::formattable<geo::Segment, char>);
static_assert(std::formattable<geo::BoundingBox, char>);
static_assert(std::formattable<geo::ClosestPairResult, char>);
static_assert(std::formattable<geo::DiameterResult, char>);
static_assert(std::formattable<geo::Triangle, char>);
static_assert(std::formattable<geo::Circle, char>);
static_assert(std::formattable<geo::DTWResult, char>);
static_assert(std::formattable<geo::GeoHashTable, char>);
static_assert(std::formattable<geo::GeoMatch, char>);
static_assert(std::formattable<geo::IcpResult, char>);
static_assert(std::formattable<geo::MinimumBoundingRectangle, char>);
static_assert(std::formattable<geo::VoronoiDiagram, char>);
static_assert(std::formattable<geo::KDTree2D, char>);
static_assert(std::formattable<geo::BestBinFirst, char>);
static_assert(std::formattable<geo::BestBinFirst::Result, char>);
static_assert(std::formattable<geo::TriangulationLocator, char>);
static_assert(std::formattable<geo::Orientation, char>);

TEST(GeometryFormat, ValueTypes) {
    const geo::Point2D p{1, 2.5};
    EXPECT_EQ(std::format("{}", p), "Point2D(1, 2.5)");
    EXPECT_EQ(std::format("{:>17}", p), "  Point2D(1, 2.5)");
    EXPECT_EQ(std::format("{}", geo::Segment{{0, 0}, {1, 1}}), "Segment(Point2D(0, 0), Point2D(1, 1))");
    EXPECT_EQ(std::format("{}", geo::BoundingBox{{0, 0}, {3, 4}}), "BoundingBox(min=Point2D(0, 0), max=Point2D(3, 4))");
    EXPECT_EQ(std::format("{}", geo::ClosestPairResult{{0, 0}, {3, 4}, 5}),
              "ClosestPairResult(a=Point2D(0, 0), b=Point2D(3, 4), distance=5)");
    EXPECT_EQ(std::format("{}", geo::DiameterResult{{0, 0}, {3, 4}, 5}),
              "DiameterResult(a=Point2D(0, 0), b=Point2D(3, 4), distance=5)");
    EXPECT_EQ(std::format("{}", geo::Triangle{0, 1, 2}), "Triangle(0, 1, 2)");
    EXPECT_EQ(std::format("{}", geo::Circle{{0, 0}, 2.5}), "Circle(center=Point2D(0, 0), radius=2.5)");
    EXPECT_EQ(std::format("{}", geo::IcpResult{0.5, {1, 2}, 3, 0.25}),
              "IcpResult(angle=0.5, translation=Point2D(1, 2), iterations=3, rmse=0.25)");
    EXPECT_EQ(std::format("{}", geo::BestBinFirst::Result{2, 7}), "BestBinFirst::Result(index=2, examined=7)");
    EXPECT_EQ(std::format("{}", geo::Orientation::CounterClockwise), "CounterClockwise");
    EXPECT_EQ(std::format("{}", static_cast<geo::Orientation>(9)), "Orientation(9)");
    EXPECT_EQ(std::format("{}", geo::GeoMatch{}), "GeoMatch(found=false, votes=0)");
}

TEST(GeometryFormat, ResultsAndIndexes) {
    const std::vector<geo::Point2D> pts{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    const auto dtw = geo::dynamic_time_warping(std::vector<double>{1, 2, 3}, std::vector<double>{1, 2, 3});
    EXPECT_EQ(std::format("{}", dtw), "DTWResult(distance=0, path_length=3, path_a=[0, 1, 2], path_b=[0, 1, 2])");
    EXPECT_EQ(std::format("{}", geo::KDTree2D(pts)), "KDTree2D(size=4)");
    EXPECT_EQ(std::format("{}", geo::BestBinFirst(pts)),
              "BestBinFirst(size=4, points=[Point2D(0, 0), Point2D(1, 0), Point2D(0, 1), Point2D(1, 1)])");
    EXPECT_EQ(std::format("{}", geo::BestBinFirst({})), "BestBinFirst(size=0, points=[])");
    const geo::MinimumBoundingRectangle r{{{0, 0}, {2, 0}, {2, 1}, {0, 1}}, 2, 1, 2};
    EXPECT_EQ(std::format("{}", r), "MinimumBoundingRectangle(width=2, height=1, area=2, corners=[Point2D(0, 0), "
                                    "Point2D(2, 0), Point2D(2, 1), Point2D(0, 1)])");
    const auto tris = geo::delaunay_triangulation(pts);
    const geo::TriangulationLocator loc(pts, tris);
    EXPECT_EQ(std::format("{}", loc), "TriangulationLocator(points=4, triangles=2, edges=5)");
    EXPECT_EQ(std::format("{}", geo::VoronoiDiagram{{{0.5, 0.5}}, {{}, {}}}), "VoronoiDiagram(vertices=1, cells=2)");
    EXPECT_EQ(std::format("{}", geo::GeoHashTable{}), "GeoHashTable(quant=0.1, bins=0)");
}
