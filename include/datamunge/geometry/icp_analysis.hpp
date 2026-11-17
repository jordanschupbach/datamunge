#pragma once
/// @file icp_analysis.hpp
/// @brief Native planar registration fixtures, checked pose application and actual-state graphics.
#include <cmath>
#include <datamunge/geometry/icp.hpp>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <format>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Return the original perturbed five-by-five grid in its original loop order.
/// Uses 3*i+.3*sin(i+j),3*j+.2*cos(i*j), i then j from zero through four.
/// Owns all 25 points; no RNG, external data, mutation or retained references.
/// @return Original finite deterministic source cloud.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::icp_example_source().size()); // 25
/// @endcode
/// @see icp_transform_points(), icp_trace()
inline std::vector<Point2D> icp_example_source() {
  std::vector<Point2D> out;
  out.reserve(25);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      out.push_back({i * 3. + .3 * std::sin(i + j), j * 3. + .2 * std::cos(i * j)});
  return out;
}
/// @brief Apply one finite planar rigid pose to a complete finite borrowed cloud.
/// Checks all input coordinates and pose before output allocation. Empty clouds
/// produce empty output after pose validation. O(N) time/owned storage, no shared
/// state/mutation/retained references. Ordinary checked double arithmetic may
/// round or overflow; no normalization or hidden fitting is performed.
/// @param points Borrowed complete finite points.
/// @param angle Finite radians, unwrapped values allowed.
/// @param translation Finite translation, default zero.
/// @return Owned transformed points in original order.
/// @throws std::invalid_argument On nonfinite input coordinates/pose.
/// @throws std::overflow_error On nonfinite transformed coordinates.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto p=dm::geometry::icp_transform_points({{1,2}},0,{3,4});
/// std::print("{} {}\n",p[0].x,p[0].y); // 4 6
/// @endcode
/// @see icp_run()
inline std::vector<Point2D> icp_transform_points(
    const std::vector<Point2D>& points,
    double                      angle,
    Point2D                     translation = {0, 0}) {
  if (!std::isfinite(angle) || !std::isfinite(translation.x) || !std::isfinite(translation.y))
    throw std::invalid_argument("ICP finite pose required");
  for (auto p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("ICP finite points required");
  const double         cs = std::cos(angle), sn = std::sin(angle);
  std::vector<Point2D> out;
  out.reserve(points.size());
  for (auto p : points) {
    const Point2D q{cs * p.x - sn * p.y + translation.x, sn * p.x + cs * p.y + translation.y};
    if (!std::isfinite(q.x) || !std::isfinite(q.y))
      throw std::overflow_error("ICP transformed coordinate range");
    out.push_back(q);
  }
  return out;
}
/// @brief Plot supplied original source, target and actual final working cloud.
/// Validates all coordinates, and final cloud/source size, before graphics.
/// Does not rerun registration or certify that an edited run came from the
/// supplied inputs. O(N+M) point/graphic storage, no retained references or mutation.
/// Axes use data ranges; aspect ratio is not an isometric geometry certificate.
/// @param source Borrowed complete original source, empty allowed.
/// @param target Borrowed complete original target, empty allowed.
/// @param run Borrowed actual final working state; aligned size must equal source size.
/// @return Owned native scatter plot, ready for titles/limits/save_svg().
/// @throws std::invalid_argument On nonfinite point coordinates or mismatched final size.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::geometry::icp_run({{0,0}},{{1,0}},1);
/// const auto p=dm::geometry::icp_alignment_plot({{0,0}},{{1,0}},r);
/// std::print("{}\n",p.to_svg().empty()); // false
/// @endcode
/// @see icp_run(), icp_step_diagram()
inline graphics::RPlot icp_alignment_plot(
    const std::vector<Point2D>& source,
    const std::vector<Point2D>& target,
    const IcpRun&               run) {
  if (source.size() != run.aligned.size())
    throw std::invalid_argument("ICP plot final size");
  for (const auto* cloud : {&source, &target, &run.aligned})
    for (auto p : *cloud)
      if (!std::isfinite(p.x) || !std::isfinite(p.y))
        throw std::invalid_argument("ICP plot finite points");
  auto                                           plot = graphics::RPlot::create();
  const std::vector<const std::vector<Point2D>*> clouds{&source, &target, &run.aligned};
  const std::vector<std::string>                 labels{"source", "target", "aligned"};
  const std::vector<graphics::RGB> colors{{148, 163, 184}, {22, 163, 74}, {220, 38, 38}};
  for (std::size_t k = 0; k < clouds.size(); ++k) {
    std::vector<double> x, y;
    x.reserve(clouds[k]->size());
    y.reserve(clouds[k]->size());
    for (auto p : *clouds[k]) {
      x.push_back(p.x);
      y.push_back(p.y);
    }
    plot.points(x, y, labels[k], colors[k], k == 1 ? 7. : 4.);
  }
  plot.x_label("x").y_label("y");
  return plot;
}
/// @brief Draw one actual captured matching/update as a native method diagram.
/// Displays actual selected target indices and error/angle/translation values;
/// it neither replays an update nor proves an edited record's mathematical origin.
/// Validates visited dimensions, indices and displayed scalar domains. A complete
/// source-point display quota precedes string/layout construction. No input
/// mutation, retained references or shared state; O(N) label work/storage.
/// @param trace Borrowed actual input/history bundle.
/// @param step Zero-based captured step, default zero.
/// @param max_points Complete displayed source count cap, default 12; zero disables.
/// @return Owned fitted native method Diagram.
/// @throws std::out_of_range On absent step.
/// @throws std::invalid_argument On visited shape/index/scalar errors.
/// @throws std::length_error On exceeded nonzero display quota.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::geometry::icp_trace({{0,0}},{{1,0}},1);
/// std::print("{}\n",dm::geometry::icp_step_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see icp_trace(), icp_alignment_plot()
inline graphics::Diagram
icp_step_diagram(const IcpTrace& trace, std::size_t step = 0, std::size_t max_points = 12) {
  if (step >= trace.steps.size())
    throw std::out_of_range("ICP absent diagram step");
  const auto& s = trace.steps[step];
  const auto  n = s.before.size();
  if (!n || s.after.size() != n || s.target_indices.size() != n || trace.source.size() != n)
    throw std::invalid_argument("ICP diagram dimensions");
  if (max_points && n > max_points)
    throw std::length_error("ICP complete display cap");
  for (auto i : s.target_indices)
    if (i >= trace.target.size())
      throw std::invalid_argument("ICP diagram target index");
  for (double x :
       {s.pre_rmse,
        s.post_fixed_rmse,
        s.post_nearest_rmse,
        s.incremental_angle,
        s.incremental_translation.x,
        s.incremental_translation.y})
    if (!std::isfinite(x))
      throw std::invalid_argument("ICP diagram finite scalars");
  if (s.pre_rmse < 0 || s.post_fixed_rmse < 0 || s.post_nearest_rmse < 0)
    throw std::invalid_argument("ICP diagram error domain");
  std::string matches;
  for (std::size_t i = 0; i < n; ++i) {
    if (i)
      matches += "\n";
    matches += std::format("source {} -> target {}", i, s.target_indices[i]);
  }
  graphics::Diagram d;
  d.node("before", {0, 0}, std::format("Before update\nRMSE {:.4g}", s.pre_rmse));
  d.node("match", {6, 0}, matches);
  d.node(
      "fit",
      {12, 0},
      std::format(
          "Rigid update\nangle {:.4g}\nshift ({:.4g}, {:.4g})",
          s.incremental_angle,
          s.incremental_translation.x,
          s.incremental_translation.y));
  d.node(
      "after",
      {18, 0},
      std::format(
          "After update\nfixed {:.4g}\nnearest {:.4g}", s.post_fixed_rmse, s.post_nearest_rmse));
  d.edge("before", "match").edge("match", "fit").edge("fit", "after");
  return d;
}
} // namespace datamunge::geometry
