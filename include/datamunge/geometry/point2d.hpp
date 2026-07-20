#pragma once

#include <cmath>

namespace datamunge::geometry {

/// @brief A point (or free vector) in the plane.
struct Point2D {
    double x{0.0};
    double y{0.0};

    bool operator==(const Point2D& other) const { return x == other.x && y == other.y; }
    bool operator!=(const Point2D& other) const { return !(*this == other); }
};

inline double squared_distance(const Point2D& a, const Point2D& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

inline double distance(const Point2D& a, const Point2D& b) { return std::sqrt(squared_distance(a, b)); }

/// @brief The z-component of (b - a) x (c - a): positive when a->b->c turns counterclockwise,
///        negative when clockwise, zero when the three points are collinear. The basic
///        orientation primitive every other algorithm in this module (convex hull, segment
///        intersection, point-in-polygon) is built from.
inline double cross(const Point2D& a, const Point2D& b, const Point2D& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

} // namespace datamunge::geometry
