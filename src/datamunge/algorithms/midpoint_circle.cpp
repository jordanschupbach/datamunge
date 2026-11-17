/// @file midpoint_circle.cpp
/// @brief Canonical midpoint circle API, producing histories and checked native contracts.

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/algorithms/midpoint_circle.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical midpoint circle production and capture.
namespace {
/// @brief Reject complete bounding geometry before any coordinate addition or capture.
/// @param cx Original x centre.
/// @param cy Original y centre.
/// @param r Radius, with negative empty convention.
/// @throws std::overflow_error On a nonrepresentable integer bounding coordinate.
void validate(int cx, int cy, int r) {
  if (r < 0)
    return;
  for (int centre : {cx, cy})
    if (std::int64_t(centre) - r < std::numeric_limits<int>::min() ||
        std::int64_t(centre) + r > std::numeric_limits<int>::max())
      throw std::overflow_error("midpoint circle coordinate range");
}
/// @brief One canonical emitting loop, optionally retaining actual transitions.
/// @param cx Centre x.
/// @param cy Centre y.
/// @param r Radius.
/// @param trace Optional owned capture target.
/// @param limit Optional capture cap, zero off.
/// @return Same-run actual points and branch counts.
/// @throws std::overflow_error On invalid coordinate geometry.
/// @throws std::length_error On output or capture admission failure.
MidpointCircleAnalysis
produce(int cx, int cy, int r, MidpointCircleTrace* trace, std::size_t limit) {
  validate(cx, cy, r);
  MidpointCircleAnalysis out;
  if (trace) {
    if (limit && limit < 3)
      throw std::length_error("midpoint circle capture inputs");
    trace->cx               = cx;
    trace->cy               = cy;
    trace->radius           = r;
    trace->captured_entries = 3;
  }
  if (r < 0)
    return out;
  int          x = r, y = 0;
  std::int64_t d = 1 - std::int64_t(r);
  while (x >= y) {
    if (out.points.max_size() - out.points.size() < 8)
      throw std::length_error("midpoint circle output geometry");
    if (trace) {
      if (trace->captured_entries > SIZE_MAX - 9 ||
          (limit && (trace->captured_entries > limit || limit - trace->captured_entries < 9)))
        throw std::length_error("midpoint circle complete capture");
      trace->captured_entries += 9;
    }
    const int                        oldx = x, oldy = y;
    const auto                       oldd  = d;
    const auto                       first = out.points.size();
    bool                             east  = d < 0;
    const std::array<CirclePoint, 8> points{
        {{cx + x, cy + y},
         {cx - x, cy + y},
         {cx + x, cy - y},
         {cx - x, cy - y},
         {cx + y, cy + x},
         {cx - y, cy + x},
         {cx + y, cy - x},
         {cx - y, cy - x}}};
    out.points.insert(out.points.end(), points.begin(), points.end());
    ++out.iterations;
    ++y;
    if (east) {
      d += 2 * std::int64_t(y) + 1;
      ++out.east_steps;
    } else {
      --x;
      d += 2 * (std::int64_t(y) - x) + 1;
      ++out.southeast_steps;
    }
    if (trace)
      trace->steps.push_back({oldx, oldy, oldd, east, x, y, d, first});
  }
  return out;
}
} // namespace
std::vector<CirclePoint> midpoint_circle(int cx, int cy, int r) {
  return produce(cx, cy, r, nullptr, 0).points;
}
MidpointCircleAnalysis midpoint_circle_analyze(int cx, int cy, int r) {
  return produce(cx, cy, r, nullptr, 0);
}
MidpointCircleTrace midpoint_circle_trace(int cx, int cy, int r, std::size_t limit) {
  MidpointCircleTrace t;
  t.analysis = produce(cx, cy, r, &t, limit);
  return t;
}
MidpointCircleStudy midpoint_circle_study(int cx, int cy, int r) {
  MidpointCircleStudy out;
  out.analysis      = midpoint_circle_analyze(cx, cy, r);
  out.unique_points = out.analysis.points;
  std::sort(out.unique_points.begin(), out.unique_points.end());
  out.unique_points.erase(
      std::unique(out.unique_points.begin(), out.unique_points.end()), out.unique_points.end());
  out.duplicate_emissions = out.analysis.points.size() - out.unique_points.size();
  out.radial_errors.reserve(out.unique_points.size());
  for (auto [x, y] : out.unique_points) {
    double error = std::abs(std::hypot(double(x) - cx, double(y) - cy) - r);
    out.radial_errors.push_back(error);
    out.maximum_radial_error = std::max(out.maximum_radial_error, error);
  }
  return out;
}
int midpoint_circle_demo_radius() {
  return 12;
}
} // namespace datamunge::algorithms
