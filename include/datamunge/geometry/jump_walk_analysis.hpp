#pragma once
/// @file jump_walk_analysis.hpp
/// @brief Preserved point-location fixture, bounded regular meshes and actual native graphics.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/jump_and_walk.hpp>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Return the original sixty-point seed7 mt19937 uniform[0,100) fixture.
/// Retains generation order, engine, distribution type and draw order. Exact
/// distribution mapping is standard-library dependent, as in the original report.
/// Owned points, no global RNG, retained references or mutation.
/// @return Sixty complete original query-mesh sites.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::jump_walk_demo_points().size()); // 60
/// @endcode
/// @see delaunay_triangulation()
inline std::vector<Point2D> jump_walk_demo_points() {
  std::mt19937                           rng(7);
  std::uniform_real_distribution<double> coord(0, 100);
  std::vector<Point2D>                   p;
  p.reserve(60);
  for (int i = 0; i < 60; ++i)
    p.push_back({coord(rng), coord(rng)});
  return p;
}
/// @brief Owned finite mesh input for controlled location experiments.
/// Points use row-major original identities; triangles are strict CCW and in
/// cell-row,cell-column,two-face order. No locator or borrowed internal pointers.
/// @par Example
/// @code
/// const auto m=dm::geometry::jump_walk_grid(1);
/// std::print("{} {}\n",m.points.size(),m.triangles.size()); // 4 2
/// @endcode
/// @see jump_walk_grid()
struct LocateMesh {
  std::vector<Point2D>  points;    ///< Owned complete original point coordinates.
  std::vector<Triangle> triangles; ///< Owned complete strict CCW faces in stable order.
};
/// @brief Generate a square integer grid with the lower-left to upper-right diagonal in each cell.
/// For side cells per axis, P=(side+1)^2 and T=2*side^2. Coordinates0..side are
/// exactly represented for side<=2^26. Each square's four corners are cocircular;
/// the fixed diagonals give a valid Delaunay triangulation without another solve.
/// Complete P+T admission and checked products precede allocation. O(P+T) owned
/// output/work, no randomness/shared state. Ceiling counts records, not bytes/RAM.
/// @param side Positive cells per axis, at most2^26.
/// @param max_records Complete P+T ceiling, default million; zero disables optional cap.
/// @return Owned complete grid and triangle list, no locator lifetime dependency.
/// @throws std::invalid_argument On zero side.
/// @throws std::length_error On representation/product/container limits or enabled ceiling.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto m=dm::geometry::jump_walk_grid(2);
/// std::print("{} {}\n",m.points.size(),m.triangles.size()); // 9 8
/// @endcode
/// @see TriangulationLocator
inline LocateMesh jump_walk_grid(std::size_t side, std::size_t max_records = 1000000) {
  if (!side)
    throw std::invalid_argument("locator grid needs cells");
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (side > (std::size_t(1) << 26) || side + 1 > maximum / (side + 1) || side > maximum / side / 2)
    throw std::length_error("locator grid shape");
  const auto p = (side + 1) * (side + 1), t = 2 * side * side;
  if (t > maximum - p || (max_records && p + t > max_records))
    throw std::length_error("locator complete grid ceiling");
  LocateMesh out;
  if (p > out.points.max_size() || t > out.triangles.max_size())
    throw std::length_error("locator grid capacity");
  out.points.reserve(p);
  out.triangles.reserve(t);
  for (std::size_t y = 0; y <= side; ++y)
    for (std::size_t x = 0; x <= side; ++x)
      out.points.push_back({double(x), double(y)});
  for (std::size_t y = 0; y < side; ++y)
    for (std::size_t x = 0; x < side; ++x) {
      const auto a = y * (side + 1) + x, b = a + 1, c = a + side + 1, d = c + 1;
      out.triangles.push_back({a, b, d});
      out.triangles.push_back({a, d, c});
    }
  return out;
}
/// @brief Diagram the first actual crossed/boundary edge of a producing trace.
/// Reads recorded face/endpoints/sign/neighbor without recomputing orientation or
/// search. Validates local visited bounds/finite coordinates and negative sign,
/// not full provenance of caller-edited histories. Constant graphic primitive count.
/// @param t Borrowed original-mesh/actual-test trace with a negative inspection.
/// @return Owned fitted four-node method Diagram of that actual decision.
/// @throws std::invalid_argument On absent negative test or invalid visited fields.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{4,4},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2},{0,2,3}};
/// std::print("{}\n",dm::geometry::jump_walk_diagram(dm::geometry::TriangulationLocator(p,f).trace({3,1},1)).to_svg().empty()); // false
/// @endcode
/// @see TriangulationLocator::trace()
inline graphics::Diagram jump_walk_diagram(const LocateTrace& t) {
  const auto it =
      std::find_if(t.tests.begin(), t.tests.end(), [](const LocateTest& e) { return e.sign < 0; });
  if (it == t.tests.end() || it->triangle < 0 || std::size_t(it->triangle) >= t.triangles.size() ||
      it->u >= t.points.size() || it->v >= t.points.size() || it->neighbor < -1 ||
      (it->neighbor >= 0 && std::size_t(it->neighbor) >= t.triangles.size()))
    throw std::invalid_argument("locator diagram actual fields");
  const auto a = t.points[it->u], b = t.points[it->v], q = t.analysis.query;
  for (const auto& p : {a, b, q})
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("locator diagram finite fields");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 4.5;
  d.node("face", {0, 0}, std::format("Actual face {}\nstep {}", it->triangle, it->step), style)
      .node(
          "edge",
          {6.3, 0},
          std::format(
              "Edge {} -> {}\n({:.4g},{:.4g}) -> ({:.4g},{:.4g})",
              it->u,
              it->v,
              a.x,
              a.y,
              b.x,
              b.y),
          style)
      .node(
          "test",
          {12.6, 0},
          std::format("Query ({:.4g},{:.4g})\nexact sign {}", q.x, q.y, it->sign),
          style)
      .node(
          "next",
          {18.9, 0},
          it->neighbor < 0 ? "Boundary reached" : std::format("Cross to\nface {}", it->neighbor),
          style)
      .edge("face", "edge")
      .edge("edge", "test")
      .edge("test", "next");
  return d;
}
/// @brief Plot a complete recorded mesh, actual entered-face centroid route and query.
/// Centroid connectors visualize entered-face order, not the geometric path of a
/// straight-line walk. Selected containment face is filled only on Located.
/// Uses square720pixel data region/equal finite padded spans on an850square canvas.
/// Complete P+T+steps admission and finite/index checks precede graphic copies;
/// no orientation/location replay or retained references. Not a trace provenance
/// validator. O(P+T+steps) graphics/validation, excluding SVG serialization.
/// @param t Borrowed complete owned mesh and actual entry trace.
/// @param max_records Complete displayed P+T+steps ceiling, default10000, zero off.
/// @return Owned native plot ready for title()/save_svg().
/// @throws std::invalid_argument On empty mesh/nonfinite/bad indices or unrepresentable padded spans.
/// @throws std::length_error On size overflow or enabled complete display ceiling.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const auto m=dm::geometry::jump_walk_grid(1);
/// const auto t=dm::geometry::TriangulationLocator(m.points,m.triangles).trace({.75,.25});
/// std::print("{}\n",dm::geometry::jump_walk_plot(t).to_svg().empty()); // false
/// @endcode
/// @see jump_walk_diagram()
inline graphics::RPlot jump_walk_plot(const LocateTrace& t, std::size_t max_records = 10000) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (t.triangles.size() > maximum - t.points.size() ||
      t.steps.size() > maximum - t.points.size() - t.triangles.size())
    throw std::length_error("locator display shape");
  if (max_records && t.points.size() + t.triangles.size() + t.steps.size() > max_records)
    throw std::length_error("locator complete display ceiling");
  if (t.points.empty() || t.triangles.empty())
    throw std::invalid_argument("locator empty display");
  double lo_x = t.analysis.query.x, hi_x = lo_x, lo_y = t.analysis.query.y, hi_y = lo_y;
  if (!std::isfinite(lo_x) || !std::isfinite(lo_y))
    throw std::invalid_argument("locator display query");
  for (const auto& p : t.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("locator display finite points");
    lo_x = std::min(lo_x, p.x);
    hi_x = std::max(hi_x, p.x);
    lo_y = std::min(lo_y, p.y);
    hi_y = std::max(hi_y, p.y);
  }
  for (const auto& f : t.triangles)
    if (f.a >= t.points.size() || f.b >= t.points.size() || f.c >= t.points.size())
      throw std::invalid_argument("locator display face indices");
  for (const auto& s : t.steps)
    if (s.triangle < 0 || std::size_t(s.triangle) >= t.triangles.size())
      throw std::invalid_argument("locator display step");
  const double      cx = std::midpoint(lo_x, hi_x), cy = std::midpoint(lo_y, hi_y);
  const long double span = std::max(
                        static_cast<long double>(hi_x) - lo_x,
                        static_cast<long double>(hi_y) - lo_y),
                    half = span > 0 ? span * .62L : 1;
  if (!std::isfinite(half) || half > std::numeric_limits<double>::max() ||
      !std::isfinite(cx - double(half)) || !std::isfinite(cx + double(half)) ||
      !std::isfinite(cy - double(half)) || !std::isfinite(cy + double(half)) ||
      cx - double(half) >= cx + double(half) || cy - double(half) >= cy + double(half))
    throw std::invalid_argument("locator display padded spans");
  auto                p = graphics::RPlot::create();
  std::vector<double> x0, y0, x1, y1, px, py;
  if (t.analysis.stop == LocateStop::Located) {
    if (t.analysis.triangle < 0 || std::size_t(t.analysis.triangle) >= t.triangles.size())
      throw std::invalid_argument("locator display result");
    const auto& f = t.triangles[t.analysis.triangle];
    for (auto i : {f.a, f.b, f.c}) {
      px.push_back(t.points[i].x);
      py.push_back(t.points[i].y);
    }
    p.polygon(px, py, {191, 219, 254}, true);
  }
  for (const auto& f : t.triangles) {
    const std::size_t ids[] = {f.a, f.b, f.c};
    for (int k = 0; k < 3; ++k) {
      const auto a = t.points[ids[k]], b = t.points[ids[(k + 1) % 3]];
      x0.push_back(a.x);
      y0.push_back(a.y);
      x1.push_back(b.x);
      y1.push_back(b.y);
    }
  }
  p.segments(x0, y0, x1, y1, {180, 190, 205}, 1);
  px.clear();
  py.clear();
  for (const auto& s : t.steps) {
    const auto& f = t.triangles[s.triangle];
    px.push_back(t.points[f.a].x / 3 + t.points[f.b].x / 3 + t.points[f.c].x / 3);
    py.push_back(t.points[f.a].y / 3 + t.points[f.b].y / 3 + t.points[f.c].y / 3);
  }
  if (px.size() > 1)
    p.line(px, py, "route", {37, 99, 235}, 3);
  if (!px.empty())
    p.points(px, py, "entries", {37, 99, 235}, 4);
  p.points({t.analysis.query.x}, {t.analysis.query.y}, "query", {220, 38, 38}, 6)
      .size(850, 850)
      .margins(95, 35, 60, 70)
      .x_limits(cx - double(half), cx + double(half))
      .y_limits(cy - double(half), cy + double(half))
      .x_label("x")
      .y_label("y")
      .title("Actual entered faces and located query");
  return p;
}
} // namespace datamunge::geometry
