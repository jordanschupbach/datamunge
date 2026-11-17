#pragma once
/// @file jarvis_analysis.hpp
/// @brief Preserved Jarvis fixture, deterministic hull-size controls and actual native graphics.
#include <datamunge/geometry/graham_analysis.hpp>
#include <datamunge/geometry/jarvis_march.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Return the original twenty-point Jarvis cloud in its historical order.
/// It is identical to the Graham report's fixture, so this delegates to the one
/// canonical stored list. Owned fixed data, no RNG, mutation or retained references.
/// @return All twenty original integer-coordinate points, including eight extremes.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto p=dm::geometry::jarvis_demo_points();
/// std::print("{} {}\n",p.size(),dm::geometry::jarvis_march(p).size()); // 20 8
/// @endcode
/// @see graham_demo_points()
inline std::vector<Point2D> jarvis_demo_points() {
  return graham_demo_points();
}
/// @brief Make a deterministic n-distinct-point control with four or n hull vertices.
/// Box mode starts with the unit square and adds distinct points on y=.5 strictly
/// inside it. Full-hull mode returns (i,i*i),0<=i<n. Restricts n<=2^26 so those
/// integer squares are exactly represented binary64 values; three distinct
/// parabola abscissas have determinant (b-a)*(c-a)*(c-b)>0. Box fractions remain
/// distinct in this range. Complete n admission precedes allocation. O(n) work/
/// owned output, no RNG/shared state/references; caps count points, not bytes/RAM.
/// @param n Complete output count, at least4 and at most2^26.
/// @param all_on_hull False(default) for box with h=4, true for parabola with h=n.
/// @param max_points Complete point ceiling, default million; zero disables optional cap.
/// @return Owned complete deterministic point family.
/// @throws std::invalid_argument On n<4.
/// @throws std::length_error On representation bound or enabled complete cap.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::geometry::jarvis_march(dm::geometry::jarvis_control_points(8)).size(),dm::geometry::jarvis_march(dm::geometry::jarvis_control_points(8,true)).size()); // 4 8
/// @endcode
/// @see jarvis_march_analyze()
inline std::vector<Point2D>
jarvis_control_points(std::size_t n, bool all_on_hull = false, std::size_t max_points = 1000000) {
  if (n < 4)
    throw std::invalid_argument("Jarvis control needs at least four points");
  if (n > (std::size_t(1) << 26) || (max_points && n > max_points))
    throw std::length_error("Jarvis complete control count");
  std::vector<Point2D> out;
  out.reserve(n);
  if (all_on_hull) {
    for (std::size_t i = 0; i < n; ++i) {
      const double x = static_cast<double>(i);
      out.push_back({x, x * x});
    }
  } else {
    out = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (std::size_t i = 0; i < n - 4; ++i)
      out.push_back({static_cast<double>(i + 1) / static_cast<double>(n - 3), .5});
  }
  return out;
}
/// @brief Draw one actual candidate test as a native operand/decision Diagram.
/// Uses recorded sorted indices, coordinates, exact sign and candidate outcome;
/// no orientation/search replay. Validates visited bounds and scalar outcome
/// consistency, not complete provenance of caller-edited histories. No mutation
/// or retained references. Constant graphic primitive/label count.
/// @param trace Borrowed producing sorted/event history.
/// @param event Actual Test-event index, default1 after the first Start.
/// @return Owned fitted concept Diagram of current/candidate/examined points and outcome.
/// @throws std::out_of_range On absent event.
/// @throws std::invalid_argument On a non-Test event, invalid visited fields or nonfinite shown points.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const auto t=dm::geometry::jarvis_march_trace({{0,0},{4,0},{4,4},{0,4},{2,2}});
/// std::print("{}\n",dm::geometry::jarvis_candidate_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see jarvis_march_trace()
inline graphics::Diagram jarvis_candidate_diagram(const JarvisTrace& trace, std::size_t event = 1) {
  if (event >= trace.events.size())
    throw std::out_of_range("Jarvis diagram event");
  const auto& e = trace.events[event];
  const auto  n = trace.sorted.size();
  if (e.action != JarvisAction::Test || e.origin >= n || e.before >= n || e.examined >= n ||
      e.after >= n || e.turn < -1 || e.turn > 1 ||
      e.after != ((e.turn < 0 || (e.turn == 0 && e.farther)) ? e.examined : e.before))
    throw std::invalid_argument("Jarvis diagram actual fields");
  for (auto i : {e.origin, e.before, e.examined})
    if (!std::isfinite(trace.sorted[i].point.x) || !std::isfinite(trace.sorted[i].point.y))
      throw std::invalid_argument("Jarvis diagram finite coordinates");
  const auto p = trace.sorted[e.origin].point, b = trace.sorted[e.before].point,
             q = trace.sorted[e.examined].point;
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 4.8;
  d.node("origin", {0, 0}, std::format("Origin #{}\n({:.6g}, {:.6g})", e.origin, p.x, p.y), style)
      .node(
          "before",
          {6.5, 0},
          std::format("Incumbent #{}\n({:.6g}, {:.6g})", e.before, b.x, b.y),
          style)
      .node(
          "examined",
          {13, 0},
          std::format("Examine #{}\n({:.6g}, {:.6g})", e.examined, q.x, q.y),
          style)
      .node(
          "outcome",
          {19.5, 0},
          std::format("Exact turn {}\nnext incumbent #{}", e.turn, e.after),
          style)
      .edge("origin", "before")
      .edge("before", "examined")
      .edge("examined", "outcome");
  return d;
}
/// @brief Plot input points and the actual Jarvis hull on native equal coordinate scales.
/// Delegates to the established shared Graham hull plotting recipe without
/// computing another hull. Complete point admission and finite/source bindings
/// precede copied display state. This is not a convexity/provenance validator.
/// The native square720pixel data region and equal coordinate spans preserve
/// geometry; later margin/size/limit edits can change that relationship. Full
/// finite binary64 core support does not imply representable padded plot spans.
/// O(n+h) validation/copied plot data, no mutation or retained input references.
/// @param points Borrowed nonempty complete finite original cloud.
/// @param analysis Borrowed actual matching-input result and original identities.
/// @param max_points Complete display point ceiling, default10000; zero disables.
/// @return Owned native hull/scatter plot ready for title/save_svg().
/// @throws std::invalid_argument On empty/count/source/nonfinite or padded-span violations.
/// @throws std::length_error On enabled complete display cap or container exhaustion.
/// @throws std::bad_alloc On graphics/display allocation failure.
/// @par Example
/// @code
/// const auto p=dm::geometry::jarvis_demo_points();
/// std::print("{}\n",dm::geometry::jarvis_hull_plot(p,dm::geometry::jarvis_march_analyze(p)).to_svg().empty()); // false
/// @endcode
/// @see graham_hull_plot(), jarvis_march_analyze()
inline graphics::RPlot jarvis_hull_plot(
    const std::vector<Point2D>& points,
    const JarvisAnalysis&       analysis,
    std::size_t                 max_points = 10000) {
  if (max_points && points.size() > max_points)
    throw std::length_error("Jarvis complete display cap");
  if (points.empty() || analysis.hull.empty() || analysis.hull.size() > points.size() ||
      analysis.input_indices.size() != analysis.hull.size())
    throw std::invalid_argument("Jarvis plot complete shape");
  for (const auto& p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("Jarvis plot finite input");
  for (std::size_t i = 0; i < analysis.hull.size(); ++i)
    if (analysis.input_indices[i] >= points.size() ||
        analysis.hull[i] != points[analysis.input_indices[i]])
      throw std::invalid_argument("Jarvis plot source binding");
  return graham_hull_plot(points, {analysis.hull, analysis.input_indices, {}})
      .title("Jarvis march: actual strict hull on equal coordinate scales");
}
} // namespace datamunge::geometry
