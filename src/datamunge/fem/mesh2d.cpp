#include <datamunge/fem/mesh2d.hpp>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::fem {

namespace {

struct CanonicalEdge {
    std::size_t u, v;
    CanonicalEdge(std::size_t a, std::size_t b) : u(std::min(a, b)), v(std::max(a, b)) {}
    bool operator==(const CanonicalEdge& other) const { return u == other.u && v == other.v; }
};
struct CanonicalEdgeHash {
    std::size_t operator()(const CanonicalEdge& e) const {
        return std::hash<std::size_t>()(e.u) ^ (std::hash<std::size_t>()(e.v) << 1);
    }
};

} // namespace

Mesh2D make_rectangular_mesh2d(double x0, double y0, double x1, double y1, std::size_t nx, std::size_t ny) {
    if (nx == 0 || ny == 0) {
        throw std::invalid_argument("make_rectangular_mesh2d: nx and ny must be positive");
    }
    if (x1 <= x0 || y1 <= y0) {
        throw std::invalid_argument("make_rectangular_mesh2d: upper bounds must exceed lower bounds");
    }

    Mesh2D mesh;
    const std::size_t nodes_x = nx + 1;
    const std::size_t nodes_y = ny + 1;
    mesh.nodes.reserve(nodes_x * nodes_y);

    const double dx = (x1 - x0) / static_cast<double>(nx);
    const double dy = (y1 - y0) / static_cast<double>(ny);

    auto node_index = [nodes_x](std::size_t i, std::size_t j) { return j * nodes_x + i; };

    for (std::size_t j = 0; j < nodes_y; ++j) {
        for (std::size_t i = 0; i < nodes_x; ++i) {
            mesh.nodes.push_back({x0 + static_cast<double>(i) * dx, y0 + static_cast<double>(j) * dy});
        }
    }

    mesh.triangles.reserve(nx * ny * 2);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const std::size_t v00 = node_index(i, j);
            const std::size_t v10 = node_index(i + 1, j);
            const std::size_t v01 = node_index(i, j + 1);
            const std::size_t v11 = node_index(i + 1, j + 1);
            // Both halves wound counterclockwise in the standard (x-right, y-up) plane.
            mesh.triangles.push_back({v00, v10, v11});
            mesh.triangles.push_back({v00, v11, v01});
        }
    }

    return mesh;
}

Mesh2D make_mesh2d_from_points(const std::vector<datamunge::geometry::Point2D>& points) {
    Mesh2D mesh;
    mesh.nodes = points;
    mesh.triangles = datamunge::geometry::delaunay_triangulation(points);
    return mesh;
}

std::vector<int> boundary_nodes(const Mesh2D& mesh) {
    std::unordered_map<CanonicalEdge, int, CanonicalEdgeHash> edge_count;
    for (const auto& t : mesh.triangles) {
        for (const auto& e : {CanonicalEdge(t.a, t.b), CanonicalEdge(t.b, t.c), CanonicalEdge(t.c, t.a)}) {
            ++edge_count[e];
        }
    }

    std::vector<int> result;
    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            result.push_back(static_cast<int>(edge.u));
            result.push_back(static_cast<int>(edge.v));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace datamunge::fem
