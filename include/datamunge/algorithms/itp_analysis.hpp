#pragma once
/// @file itp_analysis.hpp
/// @brief Native diagrams and matched-callback-axis graphics for actual ITP histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/bisection.hpp>
#include <datamunge/algorithms/itp.hpp>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Draw the three recorded ITP stages from one actual projection trial.
/// Displays owned scalar snapshots; no callback/solver replay, mutation or
/// retained references. Validates visited finite fields and bracket membership,
/// not complete provenance or all algebra in a caller-edited history. Constant
/// primitive count and label storage.
/// @param trace Borrowed actual ITP history.
/// @param step Zero-based candidate record, default0; must be a projection trial.
/// @return Owned fitted native concept Diagram.
/// @throws std::out_of_range On absent step.
/// @throws std::invalid_argument On a midpoint-only record or inconsistent shown metadata.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::itp_trace([](double x){return x*x*x-2;},0,2,{.125,0,100,0});
/// std::print("{}\n",dm::algorithms::itp_stage_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see itp_trace()
inline graphics::Diagram itp_stage_diagram(const ItpTrace& trace, std::size_t step = 0) {
  if (step >= trace.steps.size())
    throw std::out_of_range("ITP diagram step");
  const auto& s = trace.steps[step];
  if (!s.projected || !std::isfinite(s.lower) || !std::isfinite(s.upper) || !(s.lower < s.upper) ||
      !(s.lower < s.x && s.x < s.upper) || !std::isfinite(s.midpoint) ||
      !std::isfinite(s.interpolated) || !std::isfinite(s.truncated) || !std::isfinite(s.delta) ||
      !std::isfinite(s.projection_radius) || !std::isfinite(s.f_x) || s.interpolated < s.lower ||
      s.interpolated > s.upper || s.truncated < s.lower || s.truncated > s.upper || s.delta < 0 ||
      s.projection_radius < 0)
    throw std::invalid_argument("ITP diagram actual fields");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 6;
  d.node(
       "interpolate",
       {0, 0},
       std::format(
           "Interpolate: {:.6g}\nactual bracket [{:.6g}, {:.6g}]",
           s.interpolated,
           s.lower,
           s.upper),
       style)
      .node(
          "truncate",
          {8, 0},
          std::format(
              "Truncate: {:.6g}\ndelta {:.6g}; centre {:.6g}", s.truncated, s.delta, s.midpoint),
          style)
      .node(
          "project",
          {16, 0},
          std::format(
              "Evaluate: {:.6g}\nprojection radius {:.6g}; f(x) {:.6g}",
              s.x,
              s.projection_radius,
              s.f_x),
          style)
      .edge("interpolate", "truncate")
      .edge("truncate", "project");
  return d;
}
/// @brief Compare actual ITP/bisection residuals or pre-evaluation bracket half-widths.
/// Horizontal coordinates count callbacks, starting at three after both initial
/// endpoints. Requires nonempty histories with exactly two endpoint calls plus
/// their candidate records. residual=false plots wide pre-evaluation half-width;
/// true plots abs(actual callback). Vertical coordinates are explicitly
/// log10(max(value,floor)), not a logarithmic renderer setting. Complete combined
/// point admission precedes graphics. Validates displayed scalar values, not
/// callback identity/equal protocols or full history provenance; callers must
/// choose matched functions/stopping controls. No replay, mutation or retained
/// references. O(total points) validation, owned plot data and storage.
/// @param itp Borrowed actual projection/final-midpoint history.
/// @param bisection Borrowed actual midpoint history.
/// @param residual False(default) for bracket half-width, true for callback residual.
/// @param floor Finite positive display floor, default1e-18 in selected vertical units.
/// @param max_points Complete combined plot count cap, default10000; zero disables.
/// @return Owned native plot with labels/legend, ready for title/limits/save_svg().
/// @throws std::invalid_argument On histories, callback counts or invalid displayed values/floor.
/// @throws std::length_error On point/count cap overflow.
/// @throws std::overflow_error On unrepresentable double display values.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto f=[](double x){return x*x-2;};
/// const auto a=dm::algorithms::itp_trace(f,1,2);
/// const auto b=dm::algorithms::bisection_trace(f,1,2);
/// std::print("{}\n",dm::algorithms::itp_bisection_plot(a,b).to_svg().empty()); // false
/// @endcode
/// @see itp_trace(), bisection_trace()
inline graphics::RPlot itp_bisection_plot(
    const ItpTrace&       itp,
    const BisectionTrace& bisection,
    bool                  residual   = false,
    double                floor      = 1e-18,
    std::size_t           max_points = 10000) {
  const auto n = itp.steps.size(), m = bisection.steps.size();
  if (!n || !m || !std::isfinite(floor) || floor <= 0)
    throw std::invalid_argument("ITP comparison nonempty histories/floor");
  if (n > std::numeric_limits<std::size_t>::max() - m ||
      n > std::numeric_limits<std::size_t>::max() - 2 ||
      m > std::numeric_limits<std::size_t>::max() - 2)
    throw std::length_error("ITP comparison complete size");
  if (max_points && n + m > max_points)
    throw std::length_error("ITP comparison complete point cap");
  if (itp.run.evaluations != n + 2 || bisection.evaluations != m + 2)
    throw std::invalid_argument("ITP comparison endpoint-call contract");
  std::vector<double> ix, iy, bx, by;
  for (std::size_t k = 0; k < n; ++k) {
    const auto& s = itp.steps[k];
    if (!(s.lower < s.upper) || !std::isfinite(s.f_x))
      throw std::invalid_argument("ITP comparison displayed fields");
    const auto value = residual ? std::fabs(static_cast<long double>(s.f_x))
                                : (static_cast<long double>(s.upper) - s.lower) / 2;
    const auto y     = std::log10(std::max(value, static_cast<long double>(floor)));
    if (!std::isfinite(y))
      throw std::overflow_error("ITP comparison display range");
    ix.push_back(static_cast<double>(k + 3));
    iy.push_back(static_cast<double>(y));
  }
  for (std::size_t k = 0; k < m; ++k) {
    const auto& s = bisection.steps[k];
    if (!(s.lower < s.upper) || !std::isfinite(s.f_midpoint))
      throw std::invalid_argument("bisection comparison displayed fields");
    const auto value = residual ? std::fabs(static_cast<long double>(s.f_midpoint))
                                : (static_cast<long double>(s.upper) - s.lower) / 2;
    const auto y     = std::log10(std::max(value, static_cast<long double>(floor)));
    if (!std::isfinite(y))
      throw std::overflow_error("bisection comparison display range");
    bx.push_back(static_cast<double>(k + 3));
    by.push_back(static_cast<double>(y));
  }
  auto                p = graphics::RPlot::create();
  const graphics::RGB ic{20, 145, 95}, bc{210, 55, 45};
  p.line(ix, iy, "ITP", ic, 2)
      .points(ix, iy, "ITP", ic, 3)
      .line(bx, by, "bisection", bc, 2)
      .points(bx, by, "bisection", bc, 3)
      .legend({"ITP candidates", "bisection midpoints"}, {ic, bc})
      .x_label("actual callback evaluations")
      .y_label(
          residual ? "log10(max(actual residual, floor))"
                   : "log10(max(pre-evaluation half-width, floor))");
  return p;
}
} // namespace datamunge::algorithms
