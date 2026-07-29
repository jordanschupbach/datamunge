#pragma once

// Laplacian smoothing of a polygonal mesh / graph embedded in the plane.
// Each free vertex is nudged toward the centroid (umbrella average) of its
// graph neighbors:
//
//     v_i <- v_i + lambda * ( mean_{j in N(i)} v_j  -  v_i ),
//
// applied for a number of iterations. Vertices marked "fixed" (typically the
// boundary) are held in place. With lambda = 1 and one iteration, each free
// vertex jumps exactly to its neighbor centroid; smaller lambda relaxes more
// gently. Smoothing monotonically reduces the graph's Dirichlet energy
// sum_{(i,j) edges} |v_i - v_j|^2, untangling and regularizing a mesh.

#include <datamunge/geometry/point2d.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::geometry {

// One relaxation sweep is folded into `iterations`. `neighbors[i]` lists the
// indices adjacent to vertex i; `fixed[i]` pins vertex i. `lambda` in (0,1].
inline std::vector<Point2D>
laplacian_smooth(const std::vector<Point2D>&           vertices,
                 const std::vector<std::vector<int>>&  neighbors,
                 const std::vector<char>&              fixed,
                 double                                lambda     = 0.5,
                 int                                   iterations = 1) {
    std::vector<Point2D> v = vertices;
    const std::size_t    n = v.size();
    for (int it = 0; it < iterations; ++it) {
        std::vector<Point2D> next = v; // Jacobi update: read old, write new.
        for (std::size_t i = 0; i < n; ++i) {
            if (i < fixed.size() && fixed[i]) continue;
            if (neighbors[i].empty()) continue;
            double cx = 0, cy = 0;
            for (int j : neighbors[i]) { cx += v[j].x; cy += v[j].y; }
            const double inv = 1.0 / static_cast<double>(neighbors[i].size());
            cx *= inv;
            cy *= inv;
            next[i].x = v[i].x + lambda * (cx - v[i].x);
            next[i].y = v[i].y + lambda * (cy - v[i].y);
        }
        v.swap(next);
    }
    return v;
}

// Dirichlet energy sum over undirected edges (each counted once) of squared
// edge length -- the quantity Laplacian smoothing is descending.
inline double dirichlet_energy(const std::vector<Point2D>&          vertices,
                               const std::vector<std::vector<int>>& neighbors) {
    double e = 0;
    for (std::size_t i = 0; i < vertices.size(); ++i)
        for (int j : neighbors[i])
            if (static_cast<std::size_t>(j) > i) {
                const double dx = vertices[i].x - vertices[j].x;
                const double dy = vertices[i].y - vertices[j].y;
                e += dx * dx + dy * dy;
            }
    return e;
}

} // namespace datamunge::geometry
