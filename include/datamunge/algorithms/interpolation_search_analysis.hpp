#pragma once
/// @file interpolation_search_analysis.hpp
/// @brief Native original/worst-case inputs and actual interpolation-probe graphics.
#include <algorithm>
#include <datamunge/algorithms/interpolation_search.hpp>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Return one of the two original ten-element sorted search examples.
/// Owned values, no shared state, RNG or retained references.
/// @param skew False(default) for odd integers1..19, true for1..9 followed by1000.
/// @return Complete original input in its original order.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::interpolation_example_values().back()); // 19
/// @endcode
/// @see interpolation_search_trace()
inline std::vector<int> interpolation_example_values(bool skew = false) {
  return skew ? std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1000}
              : std::vector<int>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
}
/// @brief Construct a sorted missing-key-one control requiring exactly n-1 probes.
/// First n-1 values are zero, last is n. For n>=2, every guarded estimate for
/// key1 floors to lo, so the positive-slope interval advances one index each time.
/// This is a deterministic worst-case control, not a random-data expectation.
/// Complete count/int-range admission precedes allocation. O(n) work/storage,
/// no mutation, shared state or retained references.
/// @param n Complete length, at least2 and at most INT_MAX.
/// @param max_elements Complete output count cap, default million; zero disables.
/// @return Owned n-element sorted adversary.
/// @throws std::invalid_argument On n<2.
/// @throws std::length_error On int range or exceeded nonzero complete cap.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::interpolation_search(dm::algorithms::interpolation_adversary(8),1).probes); // 7
/// @endcode
/// @see interpolation_search_run()
inline std::vector<int> interpolation_adversary(std::size_t n, std::size_t max_elements = 1000000) {
  if (n < 2)
    throw std::invalid_argument("interpolation adversary length>=2");
  if (n > std::size_t(std::numeric_limits<int>::max()) || (max_elements && n > max_elements))
    throw std::length_error("interpolation complete adversary count");
  std::vector<int> a(n, 0);
  a.back() = static_cast<int>(n);
  return a;
}
/// @brief Draw a captured actual interval, exact offset and candidate as a native Diagram.
/// Validates visited scalar bounds/offset/flat metadata, not an edited history's
/// complete provenance. No input reads, search replay, mutation or retained
/// references. O(1) primitive count and bounded scalar label storage.
/// @param trace Borrowed actual probe/result bundle.
/// @param step Zero-based recorded probe, default0.
/// @return Owned fitted native concept Diagram.
/// @throws std::out_of_range On absent step.
/// @throws std::invalid_argument On inconsistent visited scalar metadata.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::interpolation_search_trace({1,3,5},3);
/// std::print("{}\n",dm::algorithms::interpolation_probe_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see interpolation_search_trace()
inline graphics::Diagram interpolation_probe_diagram(
    const InterpolationSearchTrace& trace,
    std::size_t                     step = 0) {
  if (step >= trace.probes.size())
    throw std::out_of_range("interpolation absent diagram probe");
  const auto& p = trace.probes[step];
  if (p.lo < 0 || p.hi < p.lo || p.position < p.lo || p.position > p.hi ||
      p.offset != std::size_t(p.position - p.lo) || p.low_value > p.high_value ||
      p.flat != (p.low_value == p.high_value) ||
      (p.flat ? (p.numerator || p.denominator || p.offset)
              : (!p.denominator || p.numerator > p.denominator)))
    throw std::invalid_argument("interpolation diagram bounds");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 5.4;
  d.node(
       "interval",
       {0, 0},
       std::format(
           "Actual interval [{}, {}]\nvalues [{}, {}]", p.lo, p.hi, p.low_value, p.high_value),
       style)
      .node(
          "offset",
          {8, 0},
          p.flat ? "Flat: no division"
                 : std::format(
                       "Exact floor offset {}\nnumerator {}\ndenominator {}",
                       p.offset,
                       p.numerator,
                       p.denominator),
          style)
      .node("probe", {16, 0}, std::format("Actual index {}\nvalue {}", p.position, p.value), style)
      .edge("interval", "offset")
      .edge("offset", "probe");
  return d;
}
/// @brief Plot original values and one actual captured endpoint secant/candidate.
/// Complete display quota and full sortedness check precede graphics. Display
/// validation is O(n) and is separate from search work; no search is rerun. The
/// horizontal key level and actual integer candidate are distinct when missing.
/// @param a Borrowed complete sorted original input.
/// @param key Actual requested target for the horizontal reference.
/// @param trace Borrowed actual probe bundle, scalar/source bindings checked at step.
/// @param step Zero-based probe, default0.
/// @param max_elements Complete display cap, default128; zero disables.
/// @return Owned native plot, ready for titles/limits/save_svg().
/// @throws std::out_of_range On absent step.
/// @throws std::invalid_argument On unsorted input or mismatched visited values/indices.
/// @throws std::length_error On exceeded nonzero complete display cap.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::interpolation_example_values();
/// const auto t=dm::algorithms::interpolation_search_trace(a,13);
/// std::print("{}\n",dm::algorithms::interpolation_secant_plot(a,13,t).to_svg().empty()); // false
/// @endcode
/// @see interpolation_search_trace(), interpolation_probe_diagram()
inline graphics::RPlot interpolation_secant_plot(
    const std::vector<int>&         a,
    int                             key,
    const InterpolationSearchTrace& trace,
    std::size_t                     step         = 0,
    std::size_t                     max_elements = 128) {
  if (max_elements && a.size() > max_elements)
    throw std::length_error("interpolation complete display cap");
  (void)interpolation_probe_diagram(trace, step);
  const auto& p = trace.probes[step];
  if (std::size_t(p.hi) >= a.size() || !std::is_sorted(a.begin(), a.end()) ||
      a[p.lo] != p.low_value || a[p.hi] != p.high_value || a[p.position] != p.value ||
      key < p.low_value || key > p.high_value)
    throw std::invalid_argument("interpolation plot source mismatch");
  std::vector<double> x, y;
  for (std::size_t i = 0; i < a.size(); ++i) {
    x.push_back(i);
    y.push_back(a[i]);
  }
  auto plot = graphics::RPlot::create();
  plot.points(x, y, "values", {148, 163, 184}, 6)
      .line(
          {double(p.lo), double(p.hi)},
          {double(p.low_value), double(p.high_value)},
          "secant",
          {37, 99, 235},
          2)
      .points({double(p.position)}, {double(p.value)}, "actual probe", {202, 138, 4}, 12)
      .abline_h(key, {100, 116, 139}, 1)
      .x_label("input index")
      .y_label("value");
  return plot;
}
} // namespace datamunge::algorithms
