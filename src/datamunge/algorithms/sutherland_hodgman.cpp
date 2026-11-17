/// @file sutherland_hodgman.cpp
/// @brief Actual half-plane emission kernel using shared stable native segment intersections.
#include <array>
#include <cmath>
#include <datamunge/algorithms/cohen_sutherland.hpp>
#include <datamunge/algorithms/sutherland_hodgman.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private invocation-local half-plane predicates and producer.
namespace {
/// @brief Test one actual point against the inclusive selected half-plane.
bool inside(const Point2& p, int e, const std::array<double, 4>& b) {
  return e == 0   ? p.first >= b[0]
         : e == 1 ? p.first <= b[2]
         : e == 2 ? p.second >= b[1]
                  : p.second <= b[3];
}
/// @brief Reuse the checked native widened segment clipper to compute an actual crossing endpoint.
Point2 intersection(
    const Point2&                a,
    const Point2&                b,
    int                          e,
    const std::array<double, 4>& box,
    bool                         a_inside) {
  const double          m = std::numeric_limits<double>::max();
  std::array<double, 4> half{-m, -m, m, m};
  half[e == 0 ? 0 : e == 1 ? 2 : e == 2 ? 1 : 3] = box[e == 0 ? 0 : e == 1 ? 2 : e == 2 ? 1 : 3];
  auto c                                         = cohen_sutherland_clip(
      a.first, a.second, b.first, b.second, half[0], half[1], half[2], half[3]);
  if (!c.visible)
    throw std::overflow_error("polygon crossing lost");
  return a_inside ? Point2{c.x1, c.y1} : Point2{c.x0, c.y0};
}
/// @brief Execute actual ordered edge classification and emitted vertices.
SutherlandRun execute(
    const std::vector<Point2>&   polygon,
    const std::array<double, 4>& box,
    const SutherlandOptions&     o,
    bool                         capture) {
  for (double v : box)
    if (!std::isfinite(v))
      throw std::invalid_argument("polygon finite bounds");
  if (box[0] > box[2] || box[1] > box[3])
    throw std::invalid_argument("polygon bounds order");
  for (auto [x, y] : polygon)
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::invalid_argument("polygon finite vertices");
  if (o.max_vertices && polygon.size() > o.max_vertices)
    throw std::length_error("polygon complete vertices");
  SutherlandRun r;
  r.polygon            = polygon;
  std::size_t retained = 0;
  /// @brief Bound an actual retained decision/point history increment.
  auto retain = [&](std::size_t n) {
    auto cap = std::numeric_limits<std::size_t>::max();
    if (retained > cap - n || (o.max_history_values && (retained > o.max_history_values ||
                                                        n > o.max_history_values - retained)))
      throw std::length_error("polygon history");
    retained += n;
  };
  for (int e = 0; e < 4 && !r.polygon.empty(); ++e) {
    auto input = std::move(r.polygon);
    r.polygon.clear();
    ++r.phases;
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (r.edge_visits == std::numeric_limits<std::size_t>::max() ||
          (o.max_edge_visits && r.edge_visits >= o.max_edge_visits))
        throw std::length_error("polygon edge visits");
      ++r.edge_visits;
      auto           cur = input[i], prev = input[i ? i - 1 : input.size() - 1];
      bool           cin = inside(cur, e, box), pin = inside(prev, e, box);
      SutherlandStep step{e, prev, cur, pin, cin, {}};
      /// @brief Append an actual produced point under the phase-output policy.
      auto emit = [&](Point2 p) {
        if ((o.max_vertices && r.polygon.size() >= o.max_vertices) ||
            r.emissions == std::numeric_limits<std::size_t>::max())
          throw std::length_error("polygon output");
        r.polygon.push_back(p);
        ++r.emissions;
        if (capture)
          step.emitted.push_back(p);
      };
      if (cin != pin) {
        auto p = intersection(prev, cur, e, box, pin);
        ++r.intersections;
        emit(p);
      }
      if (cin)
        emit(cur);
      if (capture) {
        retain(5 + 2 * step.emitted.size());
        r.steps.push_back(std::move(step));
      }
    }
    if (capture) {
      if (r.polygon.size() > (std::numeric_limits<std::size_t>::max() - 1) / 2)
        throw std::length_error("polygon phase history capacity");
      retain(1 + 2 * r.polygon.size());
      r.history.push_back({e, r.polygon});
    }
  }
  return r;
}
} // namespace
SutherlandRun sutherland_hodgman_run(
    const std::vector<Point2>& polygon,
    double                     xmin,
    double                     ymin,
    double                     xmax,
    double                     ymax,
    const SutherlandOptions&   options) {
  return execute(polygon, {xmin, ymin, xmax, ymax}, options, false);
}
SutherlandRun sutherland_hodgman_trace(
    const std::vector<Point2>& polygon,
    double                     xmin,
    double                     ymin,
    double                     xmax,
    double                     ymax,
    const SutherlandOptions&   options) {
  return execute(polygon, {xmin, ymin, xmax, ymax}, options, true);
}
std::vector<Point2> sutherland_hodgman_clip(
    const std::vector<Point2>& polygon,
    double                     xmin,
    double                     ymin,
    double                     xmax,
    double                     ymax) {
  return sutherland_hodgman_run(polygon, xmin, ymin, xmax, ymax).polygon;
}
} // namespace datamunge::algorithms
