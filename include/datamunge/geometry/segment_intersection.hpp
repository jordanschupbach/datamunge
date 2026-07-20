#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>

namespace datamunge::geometry {

enum class Orientation { Collinear, Clockwise, CounterClockwise };

[[nodiscard]] inline Orientation orientation(const Point2D& a, const Point2D& b, const Point2D& c) {
    const double val = cross(a, b, c);
    if (val > 0.0) return Orientation::CounterClockwise;
    if (val < 0.0) return Orientation::Clockwise;
    return Orientation::Collinear;
}

namespace detail {
// Given p, q, r already known collinear, is q within pr's bounding box (equivalently, on the
// segment pr)?
[[nodiscard]] inline bool on_segment(const Point2D& p, const Point2D& q, const Point2D& r) {
    return q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y);
}
} // namespace detail

/// @brief True if segments p1-q1 and p2-q2 intersect (including touching at an endpoint, or
///        overlapping collinear segments) -- the standard CLRS-style orientation-based test.
[[nodiscard]] inline bool segments_intersect(const Point2D& p1, const Point2D& q1, const Point2D& p2, const Point2D& q2) {
    const Orientation o1 = orientation(p1, q1, p2);
    const Orientation o2 = orientation(p1, q1, q2);
    const Orientation o3 = orientation(p2, q2, p1);
    const Orientation o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) {
        return true; // general case: p2 and q2 are on opposite sides of line p1q1, and vice versa
    }

    // Collinear special cases: an endpoint of one segment lies exactly on the other segment.
    if (o1 == Orientation::Collinear && detail::on_segment(p1, p2, q1)) return true;
    if (o2 == Orientation::Collinear && detail::on_segment(p1, q2, q1)) return true;
    if (o3 == Orientation::Collinear && detail::on_segment(p2, p1, q2)) return true;
    if (o4 == Orientation::Collinear && detail::on_segment(p2, q1, q2)) return true;

    return false;
}

/// @brief The intersection point of INFINITE lines through (p1, p2) and through (p3, p4), if
///        one exists (writes it to @p out and returns true) -- false for parallel or
///        (exactly) coincident lines. Unlike segments_intersect(), this treats both inputs as
///        unbounded lines, not segments.
[[nodiscard]] inline bool line_intersection_point(const Point2D& p1, const Point2D& p2, const Point2D& p3, const Point2D& p4,
                                                   Point2D& out) {
    const double d = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if (d == 0.0) return false;
    const double t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / d;
    out.x = p1.x + t * (p2.x - p1.x);
    out.y = p1.y + t * (p2.y - p1.y);
    return true;
}

/// @brief The (single) intersection point of segments p1-q1 and p2-q2, if one exists (writes
///        it to @p out and returns true). Returns false whenever segments_intersect() would
///        (non-intersecting segments) AND for the collinear-overlapping case that
///        segments_intersect() reports as true -- there, infinitely many points satisfy the
///        intersection, so no single Point2D can represent it.
[[nodiscard]] inline bool segment_intersection_point(const Point2D& p1, const Point2D& q1, const Point2D& p2, const Point2D& q2,
                                                      Point2D& out) {
    const double d = (p1.x - q1.x) * (p2.y - q2.y) - (p1.y - q1.y) * (p2.x - q2.x);
    if (d == 0.0) return false; // parallel, or collinear (in which case there's no single point)
    const double t = ((p1.x - p2.x) * (p2.y - q2.y) - (p1.y - p2.y) * (p2.x - q2.x)) / d;
    const double u = ((p1.x - p2.x) * (p1.y - q1.y) - (p1.y - p2.y) * (p1.x - q1.x)) / d;
    if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0) return false;
    out.x = p1.x + t * (q1.x - p1.x);
    out.y = p1.y + t * (q1.y - p1.y);
    return true;
}

} // namespace datamunge::geometry
