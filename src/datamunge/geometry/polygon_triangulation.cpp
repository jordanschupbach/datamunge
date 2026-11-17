/// @file polygon_triangulation.cpp
/// @brief Simple-boundary validation, exact ear tests and actual post-erasure observation.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/orientation.hpp>
#include <datamunge/geometry/polygon_triangulation.hpp>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
namespace datamunge::geometry {
/// @brief Shared validation and producing ear kernel.
namespace {
/// @brief Inclusive coordinate-box containment after a collinearity test.
bool on_box(const Point2D& p, const Point2D& a, const Point2D& b) {
  return p.x >= std::min(a.x, b.x) && p.x <= std::max(a.x, b.x) && p.y >= std::min(a.y, b.y) &&
         p.y <= std::max(a.y, b.y);
}
/// @brief Add retained logical entries without wrap or policy overrun.
void admit(std::size_t& n, std::size_t add, std::size_t cap) {
  if (add > std::numeric_limits<std::size_t>::max() - n || (cap && (n > cap || add > cap - n)))
    throw std::length_error("polygon triangulation history policy");
  n += add;
}
/// @brief Validate the complete boundary then execute exact ear clipping once.
PolygonTriangulationTrace
run(const std::vector<Point2D>& p, PolygonTriangulationOptions options, bool capture) {
  PolygonTriangulationTrace t;
  auto&                     r        = t.run;
  const auto                n        = p.size();
  std::size_t               retained = 0;
  for (const auto& q : p)
    if (!std::isfinite(q.x) || !std::isfinite(q.y))
      throw std::invalid_argument("triangulation finite coordinates required");
  if (n < 3)
    return t;
  std::set<std::pair<double, double>> seen;
  for (const auto& q : p)
    if (!seen.insert({q.x, q.y}).second)
      throw std::invalid_argument("triangulation duplicate vertex");
  /// @brief Admit and perform one actual exact orientation predicate.
  auto orient = [&](std::size_t a, std::size_t b, std::size_t c) {
    if (r.orientations == std::numeric_limits<std::size_t>::max() ||
        (options.max_predicates && r.orientations >= options.max_predicates))
      throw std::length_error("triangulation predicate policy");
    ++r.orientations;
    return orientation_sign(p[a], p[b], p[c]);
  };
  for (std::size_t i = 0; i < n; ++i) {
    const auto prev = (i + n - 1) % n, next = (i + 1) % n;
    if (orient(prev, i, next) == 0 && !on_box(p[i], p[prev], p[next]))
      throw std::invalid_argument("triangulation overlapping adjacent edges");
  }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      const auto in = (i + 1) % n, jn = (j + 1) % n;
      if (in == j || jn == i)
        continue;
      const int a = orient(i, in, j), b = orient(i, in, jn), c = orient(j, jn, i),
                d = orient(j, jn, in);
      if ((a * b < 0 && c * d < 0) || (a == 0 && on_box(p[j], p[i], p[in])) ||
          (b == 0 && on_box(p[jn], p[i], p[in])) || (c == 0 && on_box(p[i], p[j], p[jn])) ||
          (d == 0 && on_box(p[in], p[j], p[jn])))
        throw std::invalid_argument("triangulation self-intersection/touch");
    }
  const auto least = static_cast<std::size_t>(
      std::min_element(
          p.begin(),
          p.end(),
          [](const auto& a, const auto& b) { return std::pair(a.x, a.y) < std::pair(b.x, b.y); }) -
      p.begin());
  const int winding = orient((least + n - 1) % n, least, (least + 1) % n);
  if (winding == 0)
    throw std::invalid_argument("triangulation zero-area boundary");
  std::vector<std::size_t> ring(n);
  std::iota(ring.begin(), ring.end(), 0);
  if (n > 3 && winding < 0) {
    std::reverse(ring.begin(), ring.end());
    r.reversed = true;
  }
  if (capture) {
    admit(retained, n, options.max_trace_entries);
    t.initial_ring = ring;
  }
  while (ring.size() > 3) {
    bool       clipped = false;
    const auto m       = ring.size();
    for (std::size_t i = 0; i < m; ++i) {
      const auto          a = ring[(i + m - 1) % m], b = ring[i], c = ring[(i + 1) % m];
      PolygonEarCandidate e{m, {a, b, c}, orient(a, b, c), 0, false, 0, false};
      ++r.candidates;
      if (e.orientation > 0) {
        for (std::size_t k = 0; k < m; ++k) {
          auto q = ring[k];
          if (q == a || q == b || q == c)
            continue;
          ++e.containment_tests;
          ++r.containment_tests;
          const int u = orient(a, b, q), v = orient(b, c, q), w = orient(c, a, q);
          if (u >= 0 && v >= 0 && w >= 0) {
            e.has_blocker = true;
            e.blocker     = q;
            break;
          }
        }
        if (!e.has_blocker) {
          e.clipped = true;
          r.triangles.push_back(e.triangle);
          ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(i));
          clipped = true;
          if (capture) {
            admit(retained, 1, options.max_trace_entries);
            admit(retained, ring.size(), options.max_trace_entries);
            t.clips.push_back({e.triangle, ring});
          }
        }
      }
      if (capture) {
        admit(retained, 1, options.max_trace_entries);
        t.candidates.push_back(e);
      }
      if (clipped)
        break;
    }
    if (!clipped)
      throw std::runtime_error("validated polygon ear search stalled");
  }
  if (orient(ring[0], ring[1], ring[2]) == 0)
    throw std::runtime_error("triangulation final triangle degenerate");
  r.triangles.push_back({ring[0], ring[1], ring[2]});
  return t;
}
} // namespace
std::vector<Triangle> triangulate_polygon(const std::vector<Point2D>& p) {
  return run(p, {}, false).run.triangles;
}
PolygonTriangulationRun triangulate_polygon_run(
    const std::vector<Point2D>& p,
    PolygonTriangulationOptions options) {
  return run(p, options, false).run;
}
PolygonTriangulationTrace triangulate_polygon_trace(
    const std::vector<Point2D>& p,
    PolygonTriangulationOptions options) {
  return run(p, options, true);
}
} // namespace datamunge::geometry
