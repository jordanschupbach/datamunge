/// @file rotating_calipers.cpp
/// @brief Actual linear antipode walk with exact support-area ordering and owned real events.
#include <cmath>
#include <datamunge/geometry/orientation.hpp>
#include <datamunge/geometry/rotating_calipers.hpp>
#include <limits>
#include <random>
#include <stdexcept>
/// @brief Canonical planar diameter implementation.
namespace datamunge::geometry {
/// @brief Private checked rotating-pointer producer.
namespace {
/// @brief Execute one hull and actual rotating antipode walk.
/// @param points Borrowed complete original coordinates.
/// @param o Input/work/history policies.
/// @param capture Retain real events.
/// @return Owned hull, actual farthest pair and work.
DiameterRun solve(const std::vector<Point2D>& points, const DiameterOptions& o, bool capture) {
  if (o.max_points && points.size() > o.max_points)
    throw std::length_error("diameter: source policy");
  for (auto p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("diameter: nonfinite point");
  DiameterRun out;
  out.hull     = convex_hull(points);
  const auto h = out.hull.size();
  if (h < 2)
    throw std::invalid_argument("diameter: need two distinct points");
  bool found = false;
  /// @brief Admit the next actual event before its geometric operation.
  const auto admit = [&]() {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (out.area_tests > max - out.pair_tests || out.area_tests + out.pair_tests == max ||
        (o.max_events && out.area_tests + out.pair_tests >= o.max_events) ||
        (capture && o.max_trace_entries && out.events.size() >= o.max_trace_entries))
      throw std::length_error("diameter: event policy");
  };
  /// @brief Test one real antipodal endpoint pair.
  /// @param i First hull id.
  /// @param j Second hull id.
  const auto pair = [&](std::size_t i, std::size_t j) {
    admit();
    ++out.pair_tests;
    const auto        a = out.hull[i], b = out.hull[j];
    const long double d =
        std::hypot(static_cast<long double>(a.x) - b.x, static_cast<long double>(a.y) - b.y);
    if (!std::isfinite(d) || d > std::numeric_limits<double>::max())
      throw std::overflow_error("diameter: distance range");
    const double distance = static_cast<double>(d);
    const bool   improved = !found || distance > out.result.distance;
    if (improved) {
      found      = true;
      out.result = {a, b, distance};
    }
    if (capture)
      out.events.push_back({CaliperEventKind::PairTest, i, j, 0, 0, distance, improved});
  };
  if (h == 2) {
    pair(0, 1);
    return out;
  }
  std::size_t j = 1;
  for (std::size_t i = 0; i < h; ++i) {
    const auto next_i = (i + 1) % h;
    int        sign   = 0;
    while (true) {
      admit();
      ++out.area_tests;
      const auto next_j = (j + 1) % h;
      sign = cross_difference_sign(out.hull[i], out.hull[next_i], out.hull[next_j], out.hull[j]);
      if (capture)
        out.events.push_back(
            {sign > 0 ? CaliperEventKind::AreaAdvance : CaliperEventKind::AreaStop,
             i,
             j,
             next_j,
             sign,
             0,
             false});
      if (sign <= 0)
        break;
      j = next_j;
      ++out.pointer_advances;
      if (out.pointer_advances / h > 2)
        throw std::logic_error("diameter: antipode invariant");
    }
    pair(i, j);
    pair(next_i, j);
    if (sign == 0) {
      const auto next_j = (j + 1) % h;
      pair(i, next_j);
      pair(next_i, next_j);
    }
  }
  return out;
}
} // namespace
DiameterRun polygon_diameter_run(const std::vector<Point2D>& p, const DiameterOptions& o) {
  return solve(p, o, false);
}
DiameterRun polygon_diameter_trace(const std::vector<Point2D>& p, const DiameterOptions& o) {
  return solve(p, o, true);
}
DiameterResult polygon_diameter(const std::vector<Point2D>& p) {
  return solve(p, {}, false).result;
}
std::vector<Point2D> diameter_example_cloud(unsigned seed) {
  std::mt19937                     rng(seed);
  std::normal_distribution<double> normal(0, 1);
  std::vector<Point2D>             points;
  for (int i = 0; i < 50; ++i)
    points.push_back({normal(rng) * 3, normal(rng) * 2});
  return points;
}
} // namespace datamunge::geometry
