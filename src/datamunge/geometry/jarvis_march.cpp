/// @file jarvis_march.cpp
/// @brief Exact-sign sorted Jarvis kernel with original identities and actual candidate chronology.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/jarvis_march.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Private complete admission, checked work and actual gift wrapping.
namespace jarvis_detail {
/// @brief Increment selected work without wrapping.
/// @param count Actual mutable count.
/// @throws std::length_error On exhausted size_t.
void increment(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Jarvis work overflow");
  ++count;
}
/// @brief Admit one actual optional operation event before it executes.
/// @param events Borrowed optional sink, null for untraced execution.
/// @param cap Actual whole-record cap, zero disables.
/// @throws std::length_error On enabled cap or vector capacity exhaustion.
void admit(const std::vector<JarvisEvent>* events, std::size_t cap) {
  if (events && ((cap && events->size() >= cap) || events->size() == events->max_size()))
    throw std::length_error("Jarvis event cap");
}
/// @brief Append an already admitted actual event.
/// @param events Optional borrowed sink.
/// @param event Actual operation metadata.
/// @throws std::bad_alloc On event allocation failure.
void record(std::vector<JarvisEvent>* events, JarvisEvent event) {
  if (events)
    events->push_back(event);
}
/// @brief Execute sorting/deduplication and exact-sign candidate scans once.
/// @param points Borrowed complete finite input, never modified or retained.
/// @param options Complete point/online actual-event policies.
/// @param sorted Optional sink for actual distinct prepared records.
/// @param events Optional sink for actual Start/Test/Select events.
/// @return Owned actual strict hull/identities and selected work.
/// @throws std::invalid_argument On nonfinite input before internal copying.
/// @throws std::length_error On enabled caps or work/container overflow.
/// @throws std::bad_alloc On allocation failure.
JarvisAnalysis execute(
    const std::vector<Point2D>&    points,
    ConvexHullOptions              options,
    std::vector<HullIndexedPoint>* sorted,
    std::vector<JarvisEvent>*      events) {
  for (const auto& p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("Jarvis finite points required");
  if (options.max_points && points.size() > options.max_points)
    throw std::length_error("Jarvis complete point cap");
  JarvisAnalysis out;
  auto&          w = out.work;
  w.input_points   = points.size();
  std::vector<HullIndexedPoint> pts;
  pts.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    pts.push_back({points[i], i});
  /// @brief Actual lexicographic comparison with earliest original-index tie order.
  /// @param a First indexed point.
  /// @param b Second indexed point.
  /// @return Strict (x,y,original index) order.
  const auto lex = [&](const auto& a, const auto& b) {
    increment(w.sort_comparisons);
    if (a.point.x != b.point.x)
      return a.point.x < b.point.x;
    if (a.point.y != b.point.y)
      return a.point.y < b.point.y;
    return a.input_index < b.input_index;
  };
  std::sort(pts.begin(), pts.end(), lex);
  /// @brief Actual adjacent coordinate equivalence, coalescing signed zeros.
  /// @param a First sorted indexed point.
  /// @param b Next sorted indexed point.
  /// @return True for numeric coordinate equality.
  const auto same = [&](const auto& a, const auto& b) {
    increment(w.duplicate_comparisons);
    return a.point == b.point;
  };
  pts.erase(std::unique(pts.begin(), pts.end(), same), pts.end());
  const auto n      = pts.size();
  w.distinct_points = n;
  if (n < 3) {
    for (const auto& p : pts) {
      out.hull.push_back(p.point);
      out.input_indices.push_back(p.input_index);
    }
  } else {
    std::size_t current = 0;
    for (;;) {
      if (out.hull.size() >= n)
        throw std::logic_error("Jarvis strict-vertex cycle invariant");
      const auto  step = out.hull.size();
      const auto  seed = current + 1 == n ? 0 : current + 1;
      std::size_t next = seed;
      admit(events, options.max_events);
      out.hull.push_back(pts[current].point);
      out.input_indices.push_back(pts[current].input_index);
      increment(w.wrap_steps);
      record(events, {JarvisAction::Start, step, current, seed, 0, seed, 0, false});
      for (std::size_t q = 0; q < n; ++q) {
        if (q == current || q == seed)
          continue;
        admit(events, options.max_events);
        increment(w.orientation_tests);
        const auto before  = next;
        const int  turn    = orientation_sign(pts[current].point, pts[next].point, pts[q].point);
        bool       farther = false;
        if (turn == 0) {
          increment(w.collinear_tests);
          // At an extreme origin, two other collinear points are on the same ray.
          // Lexicographic index order is strictly monotone along that ray.
          farther = next > current ? q > next : q < next;
        }
        if (turn < 0 || (turn == 0 && farther)) {
          next = q;
          increment(w.candidate_updates);
        }
        record(events, {JarvisAction::Test, step, current, before, q, next, turn, farther});
      }
      admit(events, options.max_events);
      record(events, {JarvisAction::Select, step, current, next, 0, next, 0, false});
      if (next == 0)
        break;
      current = next;
    }
  }
  if (sorted)
    *sorted = std::move(pts);
  return out;
}
} // namespace jarvis_detail
std::vector<Point2D> jarvis_march(std::vector<Point2D> points) {
  return jarvis_detail::execute(points, {0, 0}, nullptr, nullptr).hull;
}
std::vector<Point2D> jarvis_march(std::vector<Point2D> points, long long& orientation_tests) {
  auto       out = jarvis_detail::execute(points, {0, 0}, nullptr, nullptr);
  const auto n   = out.work.orientation_tests;
  if (n > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
      orientation_tests > std::numeric_limits<long long>::max() - static_cast<long long>(n))
    throw std::overflow_error("Jarvis accumulating counter overflow");
  orientation_tests += static_cast<long long>(n);
  return std::move(out.hull);
}
JarvisAnalysis jarvis_march_analyze(const std::vector<Point2D>& points, ConvexHullOptions options) {
  return jarvis_detail::execute(points, options, nullptr, nullptr);
}
JarvisTrace jarvis_march_trace(const std::vector<Point2D>& points, ConvexHullOptions options) {
  JarvisTrace out;
  out.analysis = jarvis_detail::execute(points, options, &out.sorted, &out.events);
  return out;
}
} // namespace datamunge::geometry
