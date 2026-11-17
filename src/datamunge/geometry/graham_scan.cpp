/// @file graham_scan.cpp
/// @brief Shared checked exact-sign Graham kernel and optional actual compact stack capture.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/graham_scan.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::geometry {
/// @brief Private preparation, checked counters and actual stack execution.
namespace graham_detail {
/// @brief Checked addition of selected work or record counts.
/// @param a Current count.
/// @param b Increment.
/// @return Nonwrapping sum.
/// @throws std::length_error If size_t cannot represent the sum.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Graham work overflow");
  return a + b;
}
/// @brief Admit one whole optional event before its producing operation.
/// @param events Optional borrowed history sink; null disables capture.
/// @param cap Actual whole-record quota, zero unlimited.
/// @throws std::length_error If the next record exceeds policy or capacity.
void admit(const std::vector<GrahamEvent>* events, std::size_t cap) {
  if (events && ((cap && events->size() >= cap) || events->size() >= events->max_size()))
    throw std::length_error("Graham event quota");
}
/// @brief Append an already admitted complete event, without replaying its operation.
/// @param events Optional borrowed history sink.
/// @param e Actual event.
/// @throws std::bad_alloc On record allocation failure.
void record(std::vector<GrahamEvent>* events, GrahamEvent e) {
  if (events)
    events->push_back(e);
}
/// @brief Execute exact-sign sorting, original identities and the actual single sweep.
/// @param points Borrowed original points, never mutated or retained.
/// @param options Point and actual-event limits.
/// @param ordered Optional borrowed sink for actual unique scan order.
/// @param events Optional borrowed actual event sink.
/// @return Owned hull/identities and selected producing work.
/// @throws std::invalid_argument For nonfinite coordinates, before caps/indexed allocation.
/// @throws std::length_error On cap, counter or container exhaustion.
/// @throws std::bad_alloc On allocation failure.
GrahamAnalysis execute(
    const std::vector<Point2D>&    points,
    ConvexHullOptions              options,
    std::vector<HullIndexedPoint>* ordered,
    std::vector<GrahamEvent>*      events) {
  for (const auto& p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("Graham requires finite coordinates");
  if (options.max_points && points.size() > options.max_points)
    throw std::length_error("Graham point quota");
  GrahamAnalysis result;
  auto&          w = result.work;
  w.input_points   = points.size();
  std::vector<HullIndexedPoint> pts;
  pts.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    pts.push_back({points[i], i});
  /// @brief Actual lexicographic comparison, breaking coordinate ties by original index.
  const auto lex = [&](const auto& a, const auto& b) {
    w.lexicographic_comparisons = add(w.lexicographic_comparisons, 1);
    if (a.point.x != b.point.x)
      return a.point.x < b.point.x;
    if (a.point.y != b.point.y)
      return a.point.y < b.point.y;
    return a.input_index < b.input_index;
  };
  std::sort(pts.begin(), pts.end(), lex);
  /// @brief Actual adjacent coordinate-equivalence check, coalescing signed zeros.
  const auto same = [&](const auto& a, const auto& b) {
    w.duplicate_comparisons = add(w.duplicate_comparisons, 1);
    return a.point == b.point;
  };
  pts.erase(std::unique(pts.begin(), pts.end(), same), pts.end());
  w.distinct_points = pts.size();
  if (pts.size() < 3) {
    for (const auto& p : pts) {
      result.hull.push_back(p.point);
      result.input_indices.push_back(p.input_index);
    }
    if (ordered)
      *ordered = std::move(pts);
    return result;
  }
  std::size_t pivot = 0;
  for (std::size_t i = 1; i < pts.size(); ++i) {
    w.pivot_comparisons = add(w.pivot_comparisons, 1);
    if (pts[i].point.y < pts[pivot].point.y ||
        (pts[i].point.y == pts[pivot].point.y && pts[i].point.x < pts[pivot].point.x))
      pivot = i;
  }
  std::swap(pts[0], pts[pivot]);
  const auto p0 = pts[0].point;
  /// @brief Exact angular comparison over the closed upper half-plane, with near-to-far ray ties.
  const auto angular = [&](const auto& a, const auto& b) {
    w.angular_tests = add(w.angular_tests, 1);
    const int s     = orientation_sign(p0, a.point, b.point);
    if (s)
      return s > 0;
    if (a.point.y != b.point.y)
      return a.point.y < b.point.y;
    return a.point.x < b.point.x;
  };
  std::sort(pts.begin() + 1, pts.end(), angular);
  std::vector<std::size_t> stack;
  stack.reserve(pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i) {
    while (stack.size() >= 2) {
      admit(events, options.max_events);
      const auto a = stack[stack.size() - 2], b = stack.back();
      w.sweep_tests  = add(w.sweep_tests, 1);
      const int turn = orientation_sign(pts[a].point, pts[b].point, pts[i].point);
      record(events, {HullAction::test, i, i, a, b, turn, stack.size()});
      if (turn > 0)
        break;
      admit(events, options.max_events);
      w.pops = add(w.pops, 1);
      stack.pop_back();
      record(events, {HullAction::pop, i, b, 0, 0, 0, stack.size()});
    }
    admit(events, options.max_events);
    w.pushes = add(w.pushes, 1);
    stack.push_back(i);
    w.max_stack = std::max(w.max_stack, stack.size());
    record(events, {HullAction::push, i, i, 0, 0, 0, stack.size()});
  }
  for (auto i : stack) {
    result.hull.push_back(pts[i].point);
    result.input_indices.push_back(pts[i].input_index);
  }
  if (ordered)
    *ordered = std::move(pts);
  return result;
}
} // namespace graham_detail
std::vector<Point2D> graham_scan(std::vector<Point2D> points) {
  return graham_detail::execute(points, {0, 0}, nullptr, nullptr).hull;
}
std::vector<Point2D> graham_scan(std::vector<Point2D> points, long long& orientation_tests) {
  auto       a = graham_detail::execute(points, {0, 0}, nullptr, nullptr);
  const auto n = graham_detail::add(a.work.angular_tests, a.work.sweep_tests);
  if (n > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
      orientation_tests > std::numeric_limits<long long>::max() - static_cast<long long>(n))
    throw std::overflow_error("Graham accumulating counter overflow");
  orientation_tests += static_cast<long long>(n);
  return std::move(a.hull);
}
GrahamAnalysis graham_scan_analyze(const std::vector<Point2D>& points, ConvexHullOptions options) {
  return graham_detail::execute(points, options, nullptr, nullptr);
}
GrahamTrace graham_scan_trace(const std::vector<Point2D>& points, ConvexHullOptions options) {
  GrahamTrace t;
  t.analysis = graham_detail::execute(points, options, &t.ordered, &t.events);
  return t;
}
} // namespace datamunge::geometry
