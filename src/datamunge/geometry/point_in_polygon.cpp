/// @file point_in_polygon.cpp
/// @brief Exact predicate decisions in the historical half-open even-odd traversal.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/orientation.hpp>
#include <datamunge/geometry/point_in_polygon.hpp>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Shared complete input validation and producing edge traversal.
namespace {
/// @brief Reject a nonfinite stored query/vertex before geometric predicates.
void finite(const Point2D& p) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y))
    throw std::invalid_argument("point in polygon requires finite coordinates");
}
/// @brief Test closed coordinate bounds after an exact zero orientation.
bool bounds(const Point2D& q, const Point2D& a, const Point2D& b) {
  return q.x >= std::min(a.x, b.x) && q.x <= std::max(a.x, b.x) && q.y >= std::min(a.y, b.y) &&
         q.y <= std::max(a.y, b.y);
}
/// @brief Traverse once, optionally observing each actual robust parity/boundary decision.
PolygonRayTrace
run(const Point2D& q, const std::vector<Point2D>& v, bool capture, std::size_t cap) {
  finite(q);
  for (const auto& p : v)
    finite(p);
  if (capture && cap && v.size() > cap)
    throw std::length_error("polygon ray history cap");
  PolygonRayTrace t;
  bool            boundary = false;
  const auto      n        = v.size();
  if (!n)
    return t;
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const auto& a         = v[i];
    const auto& b         = v[j];
    const bool  straddles = (a.y > q.y) != (b.y > q.y);
    const int   orient    = orientation_sign(a, b, q);
    const bool  on        = orient == 0 && bounds(q, a, b);
    boundary |= on;
    const bool toggle = straddles && (b.y > a.y ? orient > 0 : orient < 0);
    if (toggle)
      t.inside = !t.inside;
    if (capture)
      t.edges.push_back({j, i, straddles, orient, on, toggle, t.inside});
  }
  t.location = boundary ? PolygonLocation::Boundary
                        : (t.inside ? PolygonLocation::Inside : PolygonLocation::Outside);
  return t;
}
} // namespace
bool point_in_polygon(const Point2D& p, const std::vector<Point2D>& v) {
  return run(p, v, false, 0).inside;
}
PolygonLocation classify_point_in_polygon(const Point2D& p, const std::vector<Point2D>& v) {
  return run(p, v, false, 0).location;
}
PolygonRayTrace
point_in_polygon_trace(const Point2D& p, const std::vector<Point2D>& v, std::size_t cap) {
  return run(p, v, true, cap);
}
} // namespace datamunge::geometry
