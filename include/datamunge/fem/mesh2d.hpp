#pragma once

#include <datamunge/geometry/delaunay.hpp>
#include <datamunge/geometry/point2d.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::fem {

/// @brief A 2D triangular finite-element mesh: shared vertex coordinates plus a triangle
///        connectivity list. Reuses datamunge::geometry's Point2D and (Delaunay-oriented,
///        always-counterclockwise) Triangle index type directly rather than duplicating them.
///        `nodes`/`triangles` are public fields for direct C++ construction, but -- same
///        rationale as Mesh1D::nodes -- are not exposed to language bindings; node_at()/
///        triangle_at()/num_nodes()/num_triangles() are the cross-language-safe accessors.
struct Mesh2D {
    std::vector<datamunge::geometry::Point2D> nodes;
    std::vector<datamunge::geometry::Triangle> triangles;

    [[nodiscard]] std::size_t num_nodes() const { return nodes.size(); }
    [[nodiscard]] std::size_t num_triangles() const { return triangles.size(); }
    [[nodiscard]] datamunge::geometry::Point2D node_at(std::size_t index) const { return nodes.at(index); }
    [[nodiscard]] datamunge::geometry::Triangle triangle_at(std::size_t index) const { return triangles.at(index); }
};

/// @brief Builds a structured rectangular mesh on [x0, x1] x [y0, y1]: nx * ny grid cells,
///        each split into two counterclockwise triangles, for a total of nx*ny*2 triangles and
///        (nx+1)*(ny+1) nodes, ordered row-major (x fastest, then y).
///        Throws std::invalid_argument if nx or ny is 0, or if x1 <= x0 or y1 <= y0.
[[nodiscard]] Mesh2D make_rectangular_mesh2d(double x0, double y0, double x1, double y1, std::size_t nx,
                                              std::size_t ny);

/// @brief Builds a mesh over the convex hull of an arbitrary point cloud via Delaunay
///        triangulation (datamunge::geometry::delaunay_triangulation) -- useful for
///        unstructured domains a structured grid does not fit.
[[nodiscard]] Mesh2D make_mesh2d_from_points(const std::vector<datamunge::geometry::Point2D>& points);

/// @brief Node indices lying on the mesh's outer boundary: every node incident to a boundary
///        edge (an edge belonging to exactly one triangle). Works generically for both a
///        structured rectangular mesh (the domain's four sides) and an unstructured Delaunay
///        mesh (the convex hull), returned in ascending order with no duplicates.
[[nodiscard]] std::vector<int> boundary_nodes(const Mesh2D& mesh);

} // namespace datamunge::fem
