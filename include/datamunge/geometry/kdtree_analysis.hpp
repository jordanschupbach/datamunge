#pragma once
/// @file kdtree_analysis.hpp
/// @brief Original k-d-tree fixture and checked actual-visit cloud graphics.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/kdtree.hpp>
#include <datamunge/plot/plot.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Generate the original200-point uniform[0,20) seed9001 mt19937 cloud or a complete prefix extension.
/// Retains original engine/distribution/draw order; exact mapping is standard-
/// library dependent. Whole n admission precedes point allocation. No global RNG.
/// @param n Complete point count, default original200, zero allowed.
/// @param seed Local mt19937 seed, default original9001.
/// @param max_points Complete point ceiling, default million, zero off.
/// @return Owned complete generated cloud in input order.
/// @throws std::length_error On enabled ceiling/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::kd_demo_points().size()); // 200
/// @endcode
/// @see KDTree2D
inline std::vector<Point2D>
kd_demo_points(std::size_t n = 200, unsigned seed = 9001, std::size_t max_points = 1000000) {
  std::vector<Point2D> out;
  if ((max_points && n > max_points) || n > out.max_size())
    throw std::length_error("KD complete fixture ceiling");
  out.reserve(n);
  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> u(0, 20);
  for (std::size_t i = 0; i < n; ++i)
    out.push_back({u(rng), u(rng)});
  return out;
}
/// @brief Plot the owned original cloud, actually visited nodes, selected nearest and query.
/// No nearest-query/distance replay. Validates complete display count/finite bounds
/// and visited input identities, not full provenance of caller-edited histories.
/// O(n+records) copied graphic work/storage; square720data region on850canvas
/// preserves equal coordinates. Later size/margin/limit edits may alter that.
/// @param t Borrowed actual owned nearest trace.
/// @param max_points Complete display point ceiling, default10000, zero off.
/// @return Owned native plot ready for title()/save_svg().
/// @throws std::invalid_argument On empty/nonfinite/bad visited identities or padded-span failure.
/// @throws std::length_error On enabled complete point ceiling.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::geometry::KDTree2D({{0,0},{2,0}}).nearest_trace({1,1});
/// std::print("{}\n",dm::geometry::kd_trace_plot(t).to_svg().empty()); // false
/// @endcode
/// @see KDNearestTrace
inline graphics::RPlot kd_trace_plot(const KDNearestTrace& t, std::size_t max_points = 10000) {
  if (max_points && t.points.size() > max_points)
    throw std::length_error("KD complete display ceiling");
  if (t.points.empty() || t.original_index >= t.points.size())
    throw std::invalid_argument("KD display result");
  double lo_x = t.query.x, hi_x = lo_x, lo_y = t.query.y, hi_y = lo_y;
  if (!std::isfinite(lo_x) || !std::isfinite(lo_y))
    throw std::invalid_argument("KD display query");
  std::vector<double> x, y, vx, vy;
  for (const auto& p : t.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("KD display finite points");
    lo_x = std::min(lo_x, p.x);
    hi_x = std::max(hi_x, p.x);
    lo_y = std::min(lo_y, p.y);
    hi_y = std::max(hi_y, p.y);
    x.push_back(p.x);
    y.push_back(p.y);
  }
  for (const auto& e : t.probes) {
    if (e.original_index >= t.points.size())
      throw std::invalid_argument("KD display probe identity");
    if (e.action == KDProbeAction::Visit) {
      vx.push_back(t.points[e.original_index].x);
      vy.push_back(t.points[e.original_index].y);
    }
  }
  const double      cx = std::midpoint(lo_x, hi_x), cy = std::midpoint(lo_y, hi_y);
  const long double span = std::max(
                        static_cast<long double>(hi_x) - lo_x,
                        static_cast<long double>(hi_y) - lo_y),
                    wide = span > 0 ? .65L * span : 1;
  if (!std::isfinite(wide) || wide > std::numeric_limits<double>::max())
    throw std::invalid_argument("KD display span");
  const double half = wide;
  if (!std::isfinite(cx - half) || !std::isfinite(cx + half) || !std::isfinite(cy - half) ||
      !std::isfinite(cy + half) || cx - half >= cx + half || cy - half >= cy + half)
    throw std::invalid_argument("KD padded display span");
  const auto selected = t.points[t.original_index];
  auto       p        = graphics::RPlot::create();
  p.points(x, y, "stored", {160, 160, 160}, 3)
      .points(vx, vy, "visited", {37, 99, 235}, 5)
      .points({selected.x}, {selected.y}, "nearest", {5, 150, 105}, 8)
      .points({t.query.x}, {t.query.y}, "query", {220, 38, 38}, 6)
      .size(850, 850)
      .margins(95, 35, 60, 70)
      .x_limits(cx - half, cx + half)
      .y_limits(cy - half, cy + half)
      .x_label("x")
      .y_label("y")
      .title("Actual nearest-query visits on equal coordinates");
  return p;
}
} // namespace datamunge::geometry
