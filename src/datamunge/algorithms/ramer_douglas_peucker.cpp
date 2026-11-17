/// @file ramer_douglas_peucker.cpp
/// @brief One checked iterative finite-segment RDP producer.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/ramer_douglas_peucker.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical polyline simplification implementation.
namespace datamunge::algorithms {
/// @brief Private finite-segment arithmetic and producer.
namespace {
/// @brief Check one point before any simplification.
/// @param p Point to validate.
/// @throws std::invalid_argument Nonfinite coordinate.
void validate(const RDPPoint& p) {
  if (!std::isfinite(p.first) || !std::isfinite(p.second))
    throw std::invalid_argument("RDP: nonfinite coordinate");
}
/// @brief Evaluate finite-segment distance for validated points.
/// @param p Point.
/// @param a First endpoint.
/// @param b Second endpoint.
/// @return Finite segment distance.
/// @throws std::overflow_error Distance outside double range.
double distance(const RDPPoint& p, const RDPPoint& a, const RDPPoint& b) {
  const long double dx  = static_cast<long double>(b.first) - a.first,
                    dy  = static_cast<long double>(b.second) - a.second,
                    px  = static_cast<long double>(p.first) - a.first,
                    py  = static_cast<long double>(p.second) - a.second;
  const long double len = std::hypot(dx, dy);
  long double       t   = 0;
  if (len > 0)
    t = std::clamp((px * (dx / len) + py * (dy / len)) / len, 0.L, 1.L);
  const auto d = static_cast<double>(std::hypot(px - t * dx, py - t * dy));
  if (!std::isfinite(d))
    throw std::overflow_error("RDP: distance outside double range");
  return d;
}
/// @brief Execute the shared iterative split producer.
/// @param points Finite ordered vertices, checked before work.
/// @param epsilon Nonnegative tolerance.
/// @param o Execution policies.
/// @param capture Retain actual split decisions.
/// @return Owned retained vertices and actual selected work.
RDPResult
solve(const std::vector<RDPPoint>& points, double epsilon, const RDPOptions& o, bool capture) {
  if (!std::isfinite(epsilon) || epsilon < 0)
    throw std::invalid_argument("RDP: invalid tolerance");
  if (o.max_points && points.size() > o.max_points)
    throw std::length_error("RDP: point policy");
  for (const auto& p : points)
    validate(p);
  RDPResult  r;
  const auto n = points.size();
  if (n == 0)
    return r;
  std::vector<char> keep(n, 0);
  keep[0] = keep[n - 1] = 1;
  std::vector<std::pair<std::size_t, std::size_t>> pending{{0, n - 1}};
  while (!pending.empty()) {
    const auto [lo, hi] = pending.back();
    pending.pop_back();
    if (hi - lo < 2)
      continue;
    if (capture && o.max_trace_entries && r.steps.size() >= o.max_trace_entries)
      throw std::length_error("RDP: history policy");
    double best = 0;
    auto   far  = lo;
    for (auto i = lo + 1; i < hi; ++i) {
      if (r.distance_tests == std::numeric_limits<std::size_t>::max() ||
          (o.max_distance_tests && r.distance_tests >= o.max_distance_tests))
        throw std::length_error("RDP: distance policy");
      ++r.distance_tests;
      const auto d = distance(points[i], points[lo], points[hi]);
      if (d > best) {
        best = d;
        far  = i;
      }
    }
    const bool split = best > epsilon;
    if (capture)
      r.steps.push_back({lo, hi, far, best, split});
    if (split) {
      keep[far] = 1;
      pending.push_back({far, hi});
      pending.push_back({lo, far});
    } else
      r.max_segment_error = std::max(r.max_segment_error, best);
  }
  for (std::size_t i = 0; i < n; ++i)
    if (keep[i]) {
      r.points.push_back(points[i]);
      r.indices.push_back(i);
    }
  return r;
}
} // namespace
double rdp_segment_distance(const RDPPoint& p, const RDPPoint& a, const RDPPoint& b) {
  validate(p);
  validate(a);
  validate(b);
  return distance(p, a, b);
}
RDPResult ramer_douglas_peucker_run(const std::vector<RDPPoint>& p, double e, const RDPOptions& o) {
  return solve(p, e, o, false);
}
RDPResult
ramer_douglas_peucker_trace(const std::vector<RDPPoint>& p, double e, const RDPOptions& o) {
  return solve(p, e, o, true);
}
std::vector<RDPPoint> ramer_douglas_peucker(const std::vector<RDPPoint>& p, double e) {
  return ramer_douglas_peucker_run(p, e).points;
}
std::vector<RDPPoint> rdp_waveform(std::size_t count, double step, std::size_t max_points) {
  if (!std::isfinite(step) || step <= 0)
    throw std::invalid_argument("rdp_waveform: invalid step");
  if (max_points && count > max_points)
    throw std::length_error("rdp_waveform: allocation policy");
  std::vector<RDPPoint> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double x = static_cast<double>(i) * step;
    if (!std::isfinite(x) || !std::isfinite(3.1 * x))
      throw std::overflow_error("rdp_waveform: nonfinite argument");
    out.emplace_back(x, std::sin(x) + 0.3 * std::sin(3.1 * x));
  }
  return out;
}
} // namespace datamunge::algorithms
