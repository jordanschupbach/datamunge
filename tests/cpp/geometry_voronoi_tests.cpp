#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

using datamunge::geometry::point_in_polygon;
using datamunge::geometry::Point2D;
using datamunge::geometry::voronoi_diagram;

TEST(VoronoiDiagram, InteriorPointGetsAClosedCellContainingItself) {
    // A square with its own center: the center's Voronoi cell is the diamond connecting the
    // midpoints of the square's four edges (the perpendicular bisectors between the center
    // and each corner) -- a closed, bounded quadrilateral, unlike the four corners' cells.
    const std::vector<Point2D> points{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}};
    const auto vd = voronoi_diagram(points);

    ASSERT_EQ(vd.cells.size(), points.size());
    const auto& center_cell = vd.cells[4];
    ASSERT_EQ(center_cell.size(), 4u);

    EXPECT_TRUE(point_in_polygon(points[4], center_cell));
    // Exact expected vertices: the midpoints of the square's four edges.
    for (const auto& v : center_cell) {
        const bool is_edge_midpoint = (v == Point2D{2, 0}) || (v == Point2D{4, 2}) || (v == Point2D{2, 4}) || (v == Point2D{0, 2});
        EXPECT_TRUE(is_edge_midpoint) << "unexpected vertex (" << v.x << ", " << v.y << ")";
    }
}

TEST(VoronoiDiagram, HullPointsGetOpenTwoVertexCells) {
    // The four corners are all on the convex hull, so (per this implementation's documented
    // simplification) their cells are open two-point polylines, not closed polygons.
    const std::vector<Point2D> points{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}};
    const auto vd = voronoi_diagram(points);

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vd.cells[i].size(), 2u);
    }
}

TEST(VoronoiDiagram, NumberOfCellsMatchesNumberOfPoints) {
    const std::vector<Point2D> points{{0, 0}, {5, 1}, {2, 6}, {8, 8}, {3, 3}, {7, 2}};
    const auto vd = voronoi_diagram(points);
    EXPECT_EQ(vd.cells.size(), points.size());
}

TEST(VoronoiDiagram, FewerThanThreePointsProducesNoVertices) {
    const auto vd = voronoi_diagram({{0, 0}, {1, 1}});
    EXPECT_TRUE(vd.vertices.empty());
}
