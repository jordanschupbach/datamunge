#pragma once

// The cone algorithm: identify the *surface* (boundary) points of a point cloud.
// An interior point of a dense sampling has neighbors in every direction; a
// boundary point has a whole angular sector -- a "cone" -- with no neighbors,
// because the cloud simply ends there. So a point is flagged as a surface point
// when the largest angular gap between the directions to its neighbors (within a
// search radius) exceeds a threshold. It is a simple, local, mesh-free way to
// extract the outline of a sampled shape, used in point-cloud processing and
// molecular-surface detection.

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::geometry {

// Returns, for each input point, whether it is a surface point: some angular gap
// between the directions to its neighbors within `radius` exceeds `gap_threshold`
// radians. Points with too few neighbors are treated as surface points.
inline std::vector<char> cone_surface_points(const std::vector<Point2D>& pts, double radius,
                                             double gap_threshold) {
    const double        r2 = radius * radius;
    std::vector<char>   surface(pts.size(), 0);
    std::vector<double> ang;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        ang.clear();
        for (std::size_t j = 0; j < pts.size(); ++j) {
            if (j == i) continue;
            const double dx = pts[j].x - pts[i].x, dy = pts[j].y - pts[i].y;
            if (dx * dx + dy * dy <= r2) ang.push_back(std::atan2(dy, dx));
        }
        if (ang.size() < 2) { surface[i] = 1; continue; } // isolated / edge
        std::sort(ang.begin(), ang.end());
        double max_gap = 0.0;
        for (std::size_t k = 1; k < ang.size(); ++k) max_gap = std::max(max_gap, ang[k] - ang[k - 1]);
        // wrap-around gap across +pi / -pi
        max_gap = std::max(max_gap, (ang.front() + 2 * M_PI) - ang.back());
        if (max_gap >= gap_threshold) surface[i] = 1;
    }
    return surface;
}

} // namespace datamunge::geometry
