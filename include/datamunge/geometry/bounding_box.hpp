#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

struct BoundingBox {
    Point2D min;
    Point2D max;
};

/// @brief The axis-aligned bounding box of @p points. Throws std::invalid_argument for an
///        empty point set.
[[nodiscard]] inline BoundingBox bounding_box(const std::vector<Point2D>& points) {
    if (points.empty()) {
        throw std::invalid_argument("bounding_box: points must be non-empty");
    }

    BoundingBox bb{points[0], points[0]};
    for (const auto& p : points) {
        bb.min.x = std::min(bb.min.x, p.x);
        bb.min.y = std::min(bb.min.y, p.y);
        bb.max.x = std::max(bb.max.x, p.x);
        bb.max.y = std::max(bb.max.y, p.y);
    }
    return bb;
}

} // namespace datamunge::geometry
