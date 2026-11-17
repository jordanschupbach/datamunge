#pragma once
/// @file inverse_iteration_analysis.hpp
/// @brief Original rounded/exact eigenvalue fixtures and native actual-iterate graphics.
#include <algorithm>
#include <datamunge/algorithms/inverse_iteration.hpp>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
namespace datamunge::algorithms {
/// @brief Build the original rounded-basis matrix or its exact-basis control.
/// Computes Q*diag(2,5,9)*Q^T in original i/j/k order. Rounded Q uses the
/// original four-decimal entries and is not exactly orthogonal; eigenvalues
/// therefore differ from 2,5,9. The control divides integer Q entries by three,
/// so its eigenvalues agree to floating precision. Owned fixed 3x3 output,
/// constant work/storage, no shared state or retained references.
/// @param rounded True(default) preserves original four-decimal Q; false uses thirds.
/// @return Owned finite symmetric original or control matrix.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::inverse_iteration_fixture(false);
/// std::print("{:.6f}\n",dm::algorithms::inverse_iteration(a,5.05,12).value); // 5.000000
/// @endcode
/// @see inverse_iteration_trace()
inline std::vector<std::vector<double>> inverse_iteration_fixture(bool rounded = true) {
  const double rounded_q[3][3] = {
      {.6667, -.6667, .3333}, {.6667, .3333, -.6667}, {.3333, .6667, .6667}};
  const double                     integer_q[3][3] = {{2, -2, 1}, {2, 1, -2}, {1, 2, 2}};
  const double                     d[3]            = {2, 5, 9};
  std::vector<std::vector<double>> a(3, std::vector<double>(3));
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k) {
        const double qi = rounded ? rounded_q[i][k] : integer_q[i][k] / 3;
        const double qj = rounded ? rounded_q[j][k] : integer_q[j][k] / 3;
        a[i][j] += qi * d[k] * qj;
      }
  return a;
}
/// @brief Draw one actual normalized-iterate transition using a native Diagram.
/// Shows the first two components and actual Rayleigh/residual diagnostics;
/// higher-dimensional vectors are identified by their complete dimension.
/// Validates visited sizes, chronology and shown finite fields, not complete
/// provenance of an edited trace. No solve/replay, mutation or retained references.
/// O(1) graphic primitives and label storage.
/// @param trace Borrowed producing trace.
/// @param step Destination solve index, default1, strictly between0 and steps.size().
/// @return Owned fitted native concept diagram.
/// @throws std::out_of_range On an absent transition.
/// @throws std::invalid_argument On inconsistent visited metadata or fewer than two components.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::inverse_iteration_trace({{2,0},{0,5}},2.1,{1});
/// std::print("{}\n",dm::algorithms::inverse_iteration_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see inverse_iteration_trace()
inline graphics::Diagram inverse_iteration_diagram(
    const InverseIterationTrace& trace,
    std::size_t                  step = 1) {
  if (step == 0 || step >= trace.steps.size())
    throw std::out_of_range("inverse iteration transition");
  const auto& before = trace.steps[step - 1];
  const auto& after  = trace.steps[step];
  if (before.iteration != step - 1 || after.iteration != step || before.pair.vector.size() < 2 ||
      before.pair.vector.size() != after.pair.vector.size() || !std::isfinite(after.pair.value) ||
      !std::isfinite(after.residual) || after.residual < 0)
    throw std::invalid_argument("inverse iteration diagram metadata");
  for (const auto* s : {&before, &after})
    for (std::size_t i = 0; i < 2; ++i)
      if (!std::isfinite(s->pair.vector[i]))
        throw std::invalid_argument("inverse iteration diagram entries");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 6.0;
  d.node(
       "before",
       {0, 0},
       std::format(
           "Actual x[{}], dimension {}\nfirst entries ({:.4f}, {:.4f})",
           before.iteration,
           before.pair.vector.size(),
           before.pair.vector[0],
           before.pair.vector[1]),
       style)
      .node("solve", {8, 0}, "Solve using saved shifted LU\nnormalize the solution", style)
      .node(
          "after",
          {16, 0},
          std::format(
              "Actual x[{}]: ({:.4f}, {:.4f})\nRayleigh {:.6f}; residual {:.3g}",
              after.iteration,
              after.pair.vector[0],
              after.pair.vector[1],
              after.pair.value,
              after.residual),
          style)
      .edge("before", "solve")
      .edge("solve", "after");
  return d;
}
/// @brief Plot log10 of actual residuals with an explicit positive display floor.
/// Plots log10(max(residual,floor)); the vertical axis is explicitly transformed,
/// not a claimed logarithmic renderer setting. Complete point quota across all
/// supplied traces is checked before graphics; scalar diagnostic validation costs
/// O(total points). No solves/replay, input mutation or retained references.
/// Does not validate complete provenance of caller-edited histories. Output owns
/// O(total points) plot data; copied traces are not made internally.
/// @param traces Borrowed nonempty actual traces, each with at least one step.
/// @param labels Borrowed nonempty legend labels, one per trace.
/// @param floor Finite positive display floor in original matrix units, default1e-18.
/// @param max_points Complete retained point cap, default10000; zero disables.
/// @return Owned plot with transformed residual axis and legend, ready to save.
/// @throws std::invalid_argument On labels, empty trace, invalid residual/floor or chronology.
/// @throws std::length_error On point quota, count overflow or iteration beyond exact double integers.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::inverse_iteration_trace({{2,0},{0,5}},2.1,{4});
/// std::print("{}\n",dm::algorithms::inverse_iteration_residual_plot({t},{"actual"}).to_svg().empty()); // false
/// @endcode
/// @see inverse_iteration_trace()
inline graphics::RPlot inverse_iteration_residual_plot(
    const std::vector<InverseIterationTrace>& traces,
    const std::vector<std::string>&           labels,
    double                                    floor      = 1e-18,
    std::size_t                               max_points = 10000) {
  if (traces.empty() || labels.size() != traces.size() || !std::isfinite(floor) || floor <= 0)
    throw std::invalid_argument("inverse iteration residual plot inputs");
  std::size_t points = 0;
  for (std::size_t i = 0; i < traces.size(); ++i) {
    if (traces[i].steps.empty() || labels[i].empty())
      throw std::invalid_argument("inverse iteration plot empty series");
    if (traces[i].steps.size() > std::numeric_limits<std::size_t>::max() - points)
      throw std::length_error("inverse iteration plot point count");
    points += traces[i].steps.size();
  }
  if (max_points && points > max_points)
    throw std::length_error("inverse iteration complete plot cap");
  for (const auto& trace : traces)
    for (std::size_t k = 0; k < trace.steps.size(); ++k) {
      const auto& s = trace.steps[k];
      if (s.iteration != k || !std::isfinite(s.residual) || s.residual < 0)
        throw std::invalid_argument("inverse iteration plot metadata");
      if (static_cast<long double>(s.iteration) > 9007199254740992.L)
        throw std::length_error("inverse iteration plot integer range");
    }
  auto                             plot = graphics::RPlot::create();
  const std::vector<graphics::RGB> palette{
      {35, 95, 210}, {220, 65, 40}, {25, 150, 100}, {155, 65, 180}};
  std::vector<graphics::RGB> colors;
  for (std::size_t i = 0; i < traces.size(); ++i) {
    std::vector<double> x, y;
    for (const auto& s : traces[i].steps) {
      x.push_back(static_cast<double>(s.iteration));
      y.push_back(std::log10(std::max(s.residual, floor)));
    }
    const auto color = palette[i % palette.size()];
    colors.push_back(color);
    plot.line(x, y, labels[i], color, 2).points(x, y, labels[i], color, 3);
  }
  plot.legend(labels, colors)
      .x_label("completed inverse solves")
      .y_label("log10(max(actual residual, display floor))");
  return plot;
}
} // namespace datamunge::algorithms
