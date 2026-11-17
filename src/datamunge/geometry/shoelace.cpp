/// @file shoelace.cpp
/// @brief One anchored compensated area producer with actual optional edge terms.
#include <cmath>
#include <datamunge/geometry/shoelace.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::geometry {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Evaluate actual compensated terms.
/// @param v Borrowed vertices.
/// @param o Policies.
/// @param capture Retain terms.
/// @return Owned actual area/work/history.
ShoelaceRun run(const std::vector<Point2D>& v, const ShoelaceOptions& o, bool capture) {
  const auto n = v.size();
  if (o.max_vertices && n > o.max_vertices)
    throw std::length_error("shoelace: vertices");
  for (const auto& p : v)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("shoelace: finite coordinates");
  ShoelaceRun r;
  if (n < 3)
    return r;
  if (capture && o.max_records && n > o.max_records)
    throw std::length_error("shoelace: history");
  long double       sum = 0, correction = 0;
  const long double ax = v.front().x, ay = v.front().y;
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const long double xj   = static_cast<long double>(v[j].x) - ax,
                      yj   = static_cast<long double>(v[j].y) - ay,
                      xi   = static_cast<long double>(v[i].x) - ax,
                      yi   = static_cast<long double>(v[i].y) - ay;
    const long double term = xj * yi - xi * yj, next = sum + term;
    if (!std::isfinite(term) || !std::isfinite(next))
      throw std::overflow_error("shoelace: intermediate");
    correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum;
    sum = next;
    ++r.edge_terms;
    if (capture)
      r.edges.push_back({j, i, term, (sum + correction) / 2});
  }
  const long double area = (sum + correction) / 2;
  if (!std::isfinite(area) || std::abs(area) > std::numeric_limits<double>::max())
    throw std::overflow_error("shoelace: area range");
  r.signed_area = static_cast<double>(area);
  r.area        = std::abs(r.signed_area);
  return r;
}
} // namespace
ShoelaceRun shoelace_run(const std::vector<Point2D>& v, const ShoelaceOptions& o) {
  return run(v, o, false);
}
ShoelaceRun shoelace_trace(const std::vector<Point2D>& v, const ShoelaceOptions& o) {
  return run(v, o, true);
}
double signed_polygon_area(const std::vector<Point2D>& v) {
  return shoelace_run(v).signed_area;
}
double polygon_area(const std::vector<Point2D>& v) {
  return shoelace_run(v).area;
}
std::vector<Point2D> shoelace_regular_polygon(std::size_t n) {
  if (n < 3)
    throw std::invalid_argument("shoelace: regular n");
  if (n > 1000000)
    throw std::length_error("shoelace: regular vertices");
  std::vector<Point2D> v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double angle = 2 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n);
    v.push_back({std::cos(angle), std::sin(angle)});
  }
  return v;
}
} // namespace datamunge::geometry
