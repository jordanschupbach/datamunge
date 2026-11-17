/// @file quickhull.cpp
/// @brief One iterative exact-predicate planar Quickhull producer.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/orientation.hpp>
#include <datamunge/geometry/quickhull.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical exact planar hull implementation.
namespace datamunge::geometry {
/// @brief Private iterative Quickhull tasks and producer.
namespace {
/// @brief Pending real baseline subproblem, or one inorder vertex emission.
struct Task {
  std::size_t              a;          ///< Baseline start or vertex to emit.
  std::size_t              b;          ///< Baseline end, ignored for emission.
  std::vector<std::size_t> candidates; ///< Owned actual strict-left subset.
  bool                     emit;       ///< Emit a vertex rather than splitting candidates.
};
/// @brief Run the shared exact iterative hull producer.
/// @param points Owned finite input, validated before sorting.
/// @param o Policies.
/// @param capture Retain actual candidate indices.
/// @return Owned strict hull and actual selected work.
QuickhullResult solve(std::vector<Point2D> points, const QuickhullOptions& o, bool capture) {
  if (o.max_points && points.size() > o.max_points)
    throw std::length_error("quickhull: input policy");
  for (const auto& p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("quickhull: nonfinite coordinate");
  std::sort(points.begin(), points.end(), [](const Point2D& a, const Point2D& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  points.erase(std::unique(points.begin(), points.end()), points.end());
  QuickhullResult r;
  r.points        = std::move(points);
  const auto& pts = r.points;
  const auto  n   = pts.size();
  if (n < 3) {
    r.hull = pts;
    return r;
  }
  /// @brief Admit one actual exact predicate before invoking it.
  const auto tick = [&](std::size_t& count) {
    if (r.predicates == std::numeric_limits<std::size_t>::max() ||
        (o.max_predicates && r.predicates >= o.max_predicates))
      throw std::length_error("quickhull: predicate policy");
    ++r.predicates;
    ++count;
  };
  /// @brief Execute one actual exact orientation on table indices.
  /// @param a Baseline start.
  /// @param b Baseline end.
  /// @param c Candidate.
  /// @return Exact represented-coordinate orientation sign.
  const auto orient = [&](std::size_t a, std::size_t b, std::size_t c) {
    tick(r.orientation_tests);
    return orientation_sign(pts[a], pts[b], pts[c]);
  };
  std::vector<std::size_t> above, below;
  for (std::size_t i = 0; i < n; ++i) {
    const int sign = orient(0, n - 1, i);
    if (sign > 0)
      above.push_back(i);
    else if (sign < 0)
      below.push_back(i);
  }
  std::vector<Task> pending;
  pending.push_back({n - 1, 0, std::move(below), false});
  pending.push_back({n - 1, 0, {}, true});
  pending.push_back({0, n - 1, std::move(above), false});
  r.hull.push_back(pts[0]);
  std::size_t history = 0;
  while (!pending.empty()) {
    Task task = std::move(pending.back());
    pending.pop_back();
    if (task.emit) {
      r.hull.push_back(pts[task.a]);
      continue;
    }
    if (task.candidates.empty())
      continue;
    auto far = task.candidates.front();
    for (std::size_t j = 1; j < task.candidates.size(); ++j) {
      const auto i = task.candidates[j];
      tick(r.area_comparisons);
      if (cross_difference_sign(pts[task.a], pts[task.b], pts[i], pts[far]) > 0)
        far = i;
    }
    std::vector<std::size_t> left, right;
    for (const auto i : task.candidates) {
      if (orient(task.a, far, i) > 0)
        left.push_back(i);
      else if (orient(far, task.b, i) > 0)
        right.push_back(i);
    }
    if (capture) {
      const auto cost = 1 + task.candidates.size() + left.size() + right.size();
      if (cost > std::numeric_limits<std::size_t>::max() - history ||
          (o.max_trace_entries &&
           (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
        throw std::length_error("quickhull: history policy");
      history += cost;
      r.steps.push_back({task.a, task.b, far, task.candidates, left, right});
    }
    pending.push_back({far, task.b, std::move(right), false});
    pending.push_back({far, 0, {}, true});
    pending.push_back({task.a, far, std::move(left), false});
  }
  if (r.hull.size() > 2) {
    tick(r.orientation_tests);
    if (orientation_sign(r.hull[0], r.hull[1], r.hull[2]) < 0)
      std::reverse(r.hull.begin(), r.hull.end());
  }
  return r;
}
} // namespace
QuickhullResult quickhull_run(std::vector<Point2D> p, const QuickhullOptions& o) {
  return solve(std::move(p), o, false);
}
QuickhullResult quickhull_trace(std::vector<Point2D> p, const QuickhullOptions& o) {
  return solve(std::move(p), o, true);
}
std::vector<Point2D> quickhull(std::vector<Point2D> p) {
  return quickhull_run(std::move(p)).hull;
}
} // namespace datamunge::geometry
