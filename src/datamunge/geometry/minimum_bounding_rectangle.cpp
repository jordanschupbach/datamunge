/// @file minimum_bounding_rectangle.cpp
/// @brief Canonical all-hull-edge rectangle scan with optional actual observations.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/minimum_bounding_rectangle.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Private canonical orientation scan.
namespace {
/// @brief Execute the rectangle producer and optional actual capture.
/// @param points Original finite input.
/// @param out Optional actual-state sink.
/// @param cap Capture quota, zero disables.
/// @return Actual best rectangle.
/// @throws std::invalid_argument If the hull is degenerate or inputs nonfinite.
/// @throws std::overflow_error If projection arithmetic is unrepresentable.
/// @throws std::length_error If capture/work count exceeds its limit.
MinimumBoundingRectangle
run(const std::vector<Point2D>& points, MinimumBoundingRectangleTrace* out, std::size_t cap) {
  /// @brief Admit owned capture units before appending/copying.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
      throw std::length_error("bounding rectangle: capture quota exceeded");
    out->captured_entries += amount;
  };
  for (auto& p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("bounding rectangle: finite points required");
  auto hull = convex_hull(points);
  if (hull.size() < 3)
    throw std::invalid_argument("bounding rectangle: nondegenerate hull required");
  if (out) {
    admit(points.size());
    admit(hull.size());
    out->input = points;
    out->hull  = hull;
  }
  MinimumBoundingRectangle best;
  best.area = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < hull.size(); ++i) {
    auto   a = hull[i], b = hull[(i + 1) % hull.size()];
    double dx = b.x - a.x, dy = b.y - a.y, len = std::hypot(dx, dy);
    if (!std::isfinite(len) || len == 0)
      throw std::overflow_error("bounding rectangle: edge geometry overflow");
    dx /= len;
    dy /= len;
    double lo_u = std::numeric_limits<double>::infinity(), hi_u = -lo_u, lo_v = lo_u, hi_v = -lo_u;
    for (auto p : hull) {
      double u = (p.x - a.x) * dx + (p.y - a.y) * dy, v = (p.x - a.x) * (-dy) + (p.y - a.y) * dx;
      if (!std::isfinite(u) || !std::isfinite(v))
        throw std::overflow_error("bounding rectangle: projection overflow");
      lo_u = std::min(lo_u, u);
      hi_u = std::max(hi_u, u);
      lo_v = std::min(lo_v, v);
      hi_v = std::max(hi_v, v);
      if (out) {
        if (out->projections == std::numeric_limits<std::size_t>::max())
          throw std::length_error("bounding rectangle: work count overflow");
        ++out->projections;
      }
    }
    const double width = hi_u - lo_u, height = hi_v - lo_v, area = width * height;
    if (!std::isfinite(width) || !std::isfinite(height) || !std::isfinite(area))
      throw std::overflow_error("bounding rectangle: area overflow");
    const bool selected = area < best.area;
    if (out) {
      admit(1);
      out->candidates.push_back({i, {dx, dy}, lo_u, hi_u, lo_v, hi_v, area, selected});
    }
    if (selected) {
      /// @brief Transform actual projected extents back into original coordinates.
      auto world = [&](double u, double v) {
        Point2D p{a.x + u * dx - v * dy, a.y + u * dy + v * dx};
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
          throw std::overflow_error("bounding rectangle: corner overflow");
        return p;
      };
      best = {
          {world(lo_u, lo_v), world(hi_u, lo_v), world(hi_u, hi_v), world(lo_u, hi_v)},
          width,
          height,
          area};
    }
  }
  if (out)
    admit(4);
  return best;
}
} // namespace
MinimumBoundingRectangle minimum_bounding_rectangle(const std::vector<Point2D>& p) {
  return run(p, nullptr, 0);
}
MinimumBoundingRectangleTrace minimum_bounding_rectangle_trace(
    const std::vector<Point2D>& p,
    std::size_t                 cap) {
  MinimumBoundingRectangleTrace t;
  t.result = run(p, &t, cap);
  return t;
}
std::vector<Point2D> bounding_rectangle_demo(double w, double h, double theta) {
  if (!std::isfinite(w) || !std::isfinite(h) || !std::isfinite(theta) || w <= 0 || h <= 0)
    throw std::invalid_argument(
        "bounding_rectangle_demo: positive finite dimensions and angle required");
  std::vector<Point2D> p{{0, 0}, {w, 0}, {w, h}, {0, h}};
  const auto           c = std::cos(theta), s = std::sin(theta);
  for (auto& q : p) {
    auto x = q.x * c - q.y * s, y = q.x * s + q.y * c;
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::overflow_error("bounding_rectangle_demo: coordinate overflow");
    q = {x, y};
  }
  return p;
}
} // namespace datamunge::geometry
