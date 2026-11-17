#pragma once
/// @file kabsch_analysis.hpp
/// @brief Original rigid-alignment fixture and native equal-scale projected fit graphics.
#include <cmath>
#include <datamunge/algorithms/kabsch.hpp>
#include <algorithm>
#include <datamunge/plot/plot.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Return the original six-point three-dimensional source cloud in input order.
/// Fixed owned data, no randomness or retained references.
/// @return Original source molecule-like control coordinates.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kabsch_demo_source().size()); // 6
/// @endcode
/// @see kabsch_demo_target()
inline std::vector<Point3> kabsch_demo_source() {
  return {{0, 0, 0}, {2, 0, 0}, {2, 1, 0}, {1, 2, 0}, {0, 1, 1}, {1, 1, 2}};
}
/// @brief Generate the original40-degree axis(1,1,1) target with translation(5,-2,1).
/// Rodrigues arithmetic/draw order retain the original report. Deterministic
/// alternating noise is (+n,-n,+n), n=-amplitude on even rows and +amplitude on
/// odd rows; this is not an independent Gaussian noise model. O(6) owned output.
/// @param amplitude Finite nonnegative noise amplitude, default original.02.
/// @return Complete six corresponding target points.
/// @throws std::invalid_argument On negative/nonfinite amplitude.
/// @throws std::overflow_error On nonfinite produced coordinates.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kabsch_demo_target().size()); // 6
/// @endcode
/// @see kabsch_analyze()
inline std::vector<Point3> kabsch_demo_target(double amplitude = .02) {
  if (!std::isfinite(amplitude) || amplitude < 0)
    throw std::invalid_argument("Kabsch fixture amplitude");
  const double ang = 40. * std::numbers::pi / 180., s = std::sin(ang), c = std::cos(ang),
               ax = 1 / std::sqrt(3.), ay = ax, az = ax;
  const auto          source = kabsch_demo_source();
  std::vector<Point3> target;
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto&  p   = source[i];
    const double dot = ax * p[0] + ay * p[1] + az * p[2], n = amplitude * (i % 2 ? 1. : -1.);
    Point3       q = {
        p[0] * c + (ay * p[2] - az * p[1]) * s + ax * dot * (1 - c) + 5 + n,
        p[1] * c + (az * p[0] - ax * p[2]) * s + ay * dot * (1 - c) - 2 - n,
        p[2] * c + (ax * p[1] - ay * p[0]) * s + az * dot * (1 - c) + 1 + n};
    for (auto x : q)
      if (!std::isfinite(x))
        throw std::overflow_error("Kabsch fixture overflow");
    target.push_back(q);
  }
  return target;
}
/// @brief Plot actual source/target/aligned xy projections on equal native coordinate scales.
/// No fit or transform replay: aligned points are the producing analysis output.
/// A projection may hide z residuals and never replaces three-dimensional RMSD.
/// Checks complete pair count/finite shape before graphic copies, not provenance.
/// O(n) validation/copied graphic storage, no mutation or retained references.
/// Square720pixel data region on850square canvas; later edits can change scales.
/// @param a Borrowed actual captured paired fit with aligned points.
/// @param max_pairs Complete display pair ceiling, default10000, zero disables optional cap.
/// @return Owned native RPlot ready for title()/save_svg().
/// @throws std::invalid_argument On empty/mismatched/nonfinite arrays or padded-span failure.
/// @throws std::length_error On enabled complete pair ceiling.
/// @throws std::bad_alloc On graphic allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kabsch_analyze(dm::algorithms::kabsch_demo_source(),dm::algorithms::kabsch_demo_target());
/// std::print("{}\n",dm::algorithms::kabsch_xy_plot(a).to_svg().empty()); // false
/// @endcode
/// @see KabschAnalysis
inline graphics::RPlot kabsch_xy_plot(const KabschAnalysis& a, std::size_t max_pairs = 10000) {
  const auto n = a.source.size();
  if (max_pairs && n > max_pairs)
    throw std::length_error("Kabsch complete display ceiling");
  if (!n || a.target.size() != n || a.aligned.size() != n)
    throw std::invalid_argument("Kabsch display shape");
  double              lo_x = a.source[0][0], hi_x = lo_x, lo_y = a.source[0][1], hi_y = lo_y;
  std::vector<double> sx, sy, tx, ty, ax, ay;
  for (std::size_t i = 0; i < n; ++i) {
    for (const auto& p : {a.source[i], a.target[i], a.aligned[i]}) {
      for (auto x : p)
        if (!std::isfinite(x))
          throw std::invalid_argument("Kabsch nonfinite display");
      lo_x = std::min(lo_x, p[0]);
      hi_x = std::max(hi_x, p[0]);
      lo_y = std::min(lo_y, p[1]);
      hi_y = std::max(hi_y, p[1]);
    }
    sx.push_back(a.source[i][0]);
    sy.push_back(a.source[i][1]);
    tx.push_back(a.target[i][0]);
    ty.push_back(a.target[i][1]);
    ax.push_back(a.aligned[i][0]);
    ay.push_back(a.aligned[i][1]);
  }
  const double      cx = std::midpoint(lo_x, hi_x), cy = std::midpoint(lo_y, hi_y);
  const long double span = std::max(
                        static_cast<long double>(hi_x) - lo_x,
                        static_cast<long double>(hi_y) - lo_y),
                    wide = span > 0 ? span * .65L : 1;
  if (!std::isfinite(wide) || wide > std::numeric_limits<double>::max())
    throw std::invalid_argument("Kabsch display span");
  const double half = wide;
  if (!std::isfinite(cx - half) || !std::isfinite(cx + half) || !std::isfinite(cy - half) ||
      !std::isfinite(cy + half) || cx - half >= cx + half || cy - half >= cy + half)
    throw std::invalid_argument("Kabsch padded display span");
  auto p = graphics::RPlot::create();
  p.points(sx, sy, "source", {120, 120, 120}, 5)
      .points(tx, ty, "target", {220, 38, 38}, 6)
      .points(ax, ay, "aligned", {22, 163, 74}, 3)
      .size(850, 850)
      .margins(95, 35, 60, 70)
      .x_limits(cx - half, cx + half)
      .y_limits(cy - half, cy + half)
      .x_label("x")
      .y_label("y")
      .title("Actual rigid alignment: xy projection");
  return p;
}
} // namespace datamunge::algorithms
