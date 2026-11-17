#pragma once
/// @file graham_analysis.hpp
/// @brief Preserved twenty-point Graham fixture and native equal-scale hull presentation.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <datamunge/geometry/graham_scan.hpp>
#include <datamunge/plot/plot.hpp>

namespace datamunge::geometry {
/// @brief Return the original report's fixed twenty-point cloud, including eight extremes.
/// Owns ordinary integer-valued coordinates; no random generator or dataset fetch.
/// @return Independent editable original fixture in its historical order.
/// @par Example
/// @code
/// const auto p=dm::geometry::graham_demo_points();
/// std::print("{} {}\n",p.size(),dm::geometry::graham_scan(p).size()); // 20 8
/// @endcode
/// @see graham_hull_plot()
inline std::vector<Point2D> graham_demo_points() {
  return {{0, 0},  {10, 0},  {18, 4},  {20, 12}, {15, 20}, {6, 22}, {-2, 15},
          {-4, 6}, {5, 5},   {10, 10}, {8, 15},  {3, 10},  {12, 6}, {14, 14},
          {2, 18}, {16, 10}, {7, 3},   {11, 17}, {9, 8},   {13, 11}};
}
/// @brief Draw copied input points and a supplied hull on equal coordinate scales.
/// This is presentation, not a hull validator: the supplied analysis should come
/// from these points. A square 720-pixel data region and equal coordinate spans
/// preserve geometry. Coordinates must be finite and the padded display span
/// must remain representable. Plot limits intentionally exclude general
/// full-range binary64 geometry even though the hull kernel supports it.
/// Empty inputs are rejected. Point/line outputs are drawn without a filled
/// polygon. No caller references survive; editing size/margins/limits may change
/// equal scaling, so preserve their relationship when editing the returned plot.
/// @param points Nonempty finite cloud to display, not mutated.
/// @param analysis Actual hull result corresponding to the same cloud.
/// @return Owned editable native plot with hull outline/fill and input markers.
/// @throws std::invalid_argument For empty/nonfinite input or unrepresentable display span.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On container exhaustion.
/// @par Example
/// @code
/// const auto p=dm::geometry::graham_demo_points();
/// dm::geometry::graham_hull_plot(p,dm::geometry::graham_scan_analyze(p)).save_svg("build/graham.svg");
/// @endcode
/// @see graham_scan_analyze(), graham_demo_points()
inline graphics::RPlot graham_hull_plot(
    const std::vector<Point2D>& points,
    const GrahamAnalysis&       analysis) {
  if (points.empty())
    throw std::invalid_argument("Graham plot needs points");
  double              xmin = points[0].x, xmax = xmin, ymin = points[0].y, ymax = ymin;
  std::vector<double> x, y, hx, hy;
  for (auto p : points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("Graham plot needs finite points");
    x.push_back(p.x);
    y.push_back(p.y);
    xmin = std::min(xmin, p.x);
    xmax = std::max(xmax, p.x);
    ymin = std::min(ymin, p.y);
    ymax = std::max(ymax, p.y);
  }
  for (auto p : analysis.hull) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("Graham plot needs finite hull");
    hx.push_back(p.x);
    hy.push_back(p.y);
  }
  const double span = std::max({xmax - xmin, ymax - ymin, 1.});
  const double cx = xmin / 2 + xmax / 2, cy = ymin / 2 + ymax / 2, half = .75 * span;
  if (!std::isfinite(half) || !std::isfinite(cx - half) || !std::isfinite(cx + half) ||
      !std::isfinite(cy - half) || !std::isfinite(cy + half) || cx - half == cx + half ||
      cy - half == cy + half)
    throw std::invalid_argument("Graham plot span is not representable");
  auto p = graphics::RPlot::create();
  if (hx.size() >= 3)
    p.polygon(hx, hy, {191, 219, 254}, true);
  else if (hx.size() == 2)
    p.line(hx, hy, "hull segment", {37, 99, 235}, 2);
  p.points(x, y, "input points", {30, 41, 59}, 4)
      .points(hx, hy, "hull vertices", {220, 38, 38}, 6)
      .size(850, 850)
      .margins(95, 35, 60, 70)
      .x_limits(cx - half, cx + half)
      .y_limits(cy - half, cy + half)
      .title("Graham scan: strict hull on equal coordinate scales")
      .x_label("x")
      .y_label("y");
  return p;
}
} // namespace datamunge::geometry
