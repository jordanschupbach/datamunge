/// @file segment_intersection.cpp
/// @brief One exact four-predicate classifier and optional approximate determinant-quotient construction.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/segment_intersection.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::geometry {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Compare finite coordinates lexicographically, respecting represented equality.
/// @param a First point.
/// @param b Second point.
/// @return Whether a precedes b.
bool less(const Point2D& a, const Point2D& b) {
  return a.x != b.x ? a.x < b.x : a.y < b.y;
}
/// @brief Closed box membership for a point already known collinear.
/// @param p First segment endpoint.
/// @param x Candidate.
/// @param q Second segment endpoint.
/// @return Actual exact coordinate-bound decision.
bool within(const Point2D& p, const Point2D& x, const Point2D& q) {
  return x.x >= std::min(p.x, q.x) && x.x <= std::max(p.x, q.x) && x.y >= std::min(p.y, q.y) &&
         x.y <= std::max(p.y, q.y);
}
/// @brief Construct a finite approximate point at an actual widened line parameter.
/// @param p First finite endpoint.
/// @param q Second finite endpoint.
/// @param t Widened finite interpolation/extrapolation parameter.
/// @return Owned finite double point.
Point2D point(const Point2D& p, const Point2D& q, long double t) {
  const long double x = std::lerp(static_cast<long double>(p.x), static_cast<long double>(q.x), t),
                    y = std::lerp(static_cast<long double>(p.y), static_cast<long double>(q.y), t);
  if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > std::numeric_limits<double>::max() ||
      std::abs(y) > std::numeric_limits<double>::max())
    throw std::overflow_error("line intersection: finite coordinate range");
  return {static_cast<double>(x), static_cast<double>(y)};
}
/// @brief Execute exact signs and actual contact branches once, with optional proper-point construction.
/// @param p First segment start.
/// @param q First segment end.
/// @param r Second segment start.
/// @param s Second segment end.
/// @param construct Construct approximate proper-crossing coordinates.
/// @return Owned actual fixed-size producing record.
SegmentIntersectionRun
classify(const Point2D& p, const Point2D& q, const Point2D& r, const Point2D& s, bool construct) {
  SegmentIntersectionRun out;
  out.orientations = {
      orientation_sign(p, q, r),
      orientation_sign(p, q, s),
      orientation_sign(r, s, p),
      orientation_sign(r, s, q)};
  out.orientation_tests = 4;
  const auto& o         = out.orientations;
  if (o[0] * o[1] < 0 && o[2] * o[3] < 0) {
    out.contact = SegmentContact::Point;
    if (construct) {
      out.parameter = std::clamp(cross_difference_ratio(p, r, s, r, p, q, s, r), 0.L, 1.L);
      out.first = out.second = point(p, q, out.parameter);
      out.point_constructed  = true;
    }
    return out;
  }
  if (o[0] == 0 && o[1] == 0 && o[2] == 0 && o[3] == 0) {
    const auto lo = less(p, q) ? p : q, hi = less(p, q) ? q : p, left = less(r, s) ? r : s,
               right = less(r, s) ? s : r;
    const auto begin = less(lo, left) ? left : lo, end = less(hi, right) ? hi : right;
    ++out.bounding_tests;
    if (less(end, begin))
      return out;
    out.contact           = less(begin, end) ? SegmentContact::Overlap : SegmentContact::Point;
    out.first             = begin;
    out.second            = end;
    out.point_constructed = true;
    return out;
  }
  const std::array<Point2D, 4> candidates{r, s, p, q};
  for (std::size_t i = 0; i < 4; ++i)
    if (o[i] == 0) {
      ++out.bounding_tests;
      if (i < 2 ? within(p, candidates[i], q) : within(r, candidates[i], s)) {
        out.contact = SegmentContact::Point;
        out.first = out.second = candidates[i];
        out.point_constructed  = true;
        return out;
      }
    }
  return out;
}
} // namespace
Orientation orientation(const Point2D& a, const Point2D& b, const Point2D& c) {
  const auto sign = orientation_sign(a, b, c);
  return sign > 0   ? Orientation::CounterClockwise
         : sign < 0 ? Orientation::Clockwise
                    : Orientation::Collinear;
}
SegmentIntersectionRun
segment_intersection_run(const Point2D& p, const Point2D& q, const Point2D& r, const Point2D& s) {
  return classify(p, q, r, s, true);
}
SegmentIntersectionRun
segment_intersection_trace(const Point2D& p, const Point2D& q, const Point2D& r, const Point2D& s) {
  return classify(p, q, r, s, true);
}
bool segments_intersect(const Point2D& p, const Point2D& q, const Point2D& r, const Point2D& s) {
  return classify(p, q, r, s, false).contact != SegmentContact::None;
}
bool line_intersection_point(
    const Point2D& p,
    const Point2D& q,
    const Point2D& r,
    const Point2D& s,
    Point2D&       out) {
  if (!cross_difference_sign(p, q, s, r))
    return false;
  const auto t     = cross_difference_ratio(p, r, s, r, p, q, s, r);
  const auto value = point(p, q, t);
  out              = value;
  return true;
}
bool segment_intersection_point(
    const Point2D& p,
    const Point2D& q,
    const Point2D& r,
    const Point2D& s,
    Point2D&       out) {
  auto value = classify(p, q, r, s, true);
  if (value.contact != SegmentContact::Point)
    return false;
  out = value.first;
  return true;
}
} // namespace datamunge::geometry
