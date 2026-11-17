/// @file clipped_voronoi.cpp
/// @brief Actual bounded nearest-site half-plane clipping and equal-scale native plots.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/clipped_voronoi.hpp>
#include <random>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Produce cells and optional actual intermediate polygons by pairwise bisector clipping.
ClippedVoronoi
clipped_voronoi(const std::vector<Point2D>& sites, VoronoiBox box, std::size_t max_records) {
  for (double v : {box.xmin, box.xmax, box.ymin, box.ymax})
    if (!std::isfinite(v) || std::abs(v) > 1e100)
      throw std::invalid_argument("Voronoi bounds");
  if (box.xmin >= box.xmax || box.ymin >= box.ymax)
    throw std::invalid_argument("Voronoi viewport");
  if (sites.size() > 1000)
    throw std::length_error("Voronoi sites");
  for (std::size_t i = 0; i < sites.size(); ++i) {
    auto p = sites[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || std::abs(p.x) > 1e100 ||
        std::abs(p.y) > 1e100)
      throw std::invalid_argument("Voronoi point");
    for (std::size_t j = 0; j < i; ++j)
      if (p.x == sites[j].x && p.y == sites[j].y)
        throw std::invalid_argument("Voronoi duplicate");
  }
  ClippedVoronoi r;
  r.sites              = sites;
  r.box                = box;
  std::size_t captured = 0;
  for (std::size_t i = 0; i < sites.size(); ++i) {
    std::vector<Point2D> poly{
        {box.xmin, box.ymin}, {box.xmax, box.ymin}, {box.xmax, box.ymax}, {box.xmin, box.ymax}};
    for (std::size_t j = 0; j < sites.size(); ++j)
      if (j != i) {
        ++r.halfplanes;
        long double ax = 2 * (static_cast<long double>(sites[j].x) - sites[i].x),
                    ay = 2 * (static_cast<long double>(sites[j].y) - sites[i].y);
        long double c  = (static_cast<long double>(sites[j].x) - sites[i].x) *
                            (static_cast<long double>(sites[j].x) + sites[i].x) +
                        (static_cast<long double>(sites[j].y) - sites[i].y) *
                            (static_cast<long double>(sites[j].y) + sites[i].y);
        // Signed bisector value in widened arithmetic; nonpositive means owned by site i.
        auto                 value = [&](Point2D p) { return ax * p.x + ay * p.y - c; };
        std::vector<Point2D> out;
        if (!poly.empty()) {
          auto        p  = poly.back();
          long double fp = value(p);
          for (auto q : poly) {
            if (++r.edge_tests > 20000000)
              throw std::length_error("Voronoi edge tests");
            long double fq = value(q);
            if ((fp <= 0) != (fq <= 0)) {
              long double t = fp / (fp - fq);
              Point2D     v{
                  double(static_cast<long double>(p.x) + t * (static_cast<long double>(q.x) - p.x)),
                  double(
                      static_cast<long double>(p.y) + t * (static_cast<long double>(q.y) - p.y))};
              if (!std::isfinite(v.x) || !std::isfinite(v.y))
                throw std::overflow_error("Voronoi intersection");
              out.push_back(v);
            }
            if (fq <= 0)
              out.push_back(q);
            p  = q;
            fp = fq;
          }
        }
        poly = std::move(out);
        if (r.steps.size() < max_records) {
          if (poly.size() > 1000000 - captured)
            throw std::length_error("Voronoi capture vertices");
          captured += poly.size();
          r.steps.push_back({i, j, poly});
        }
      }
    r.cells.push_back(std::move(poly));
  }
  return r;
}
/// @brief Draw the actual clipped polygons with equal coordinate spans and square plotting area.
graphics::RPlot clipped_voronoi_plot(const ClippedVoronoi& r) {
  auto p = graphics::RPlot::create();
  for (auto& cell : r.cells) {
    std::vector<double> x, y;
    for (auto q : cell) {
      x.push_back(q.x);
      y.push_back(q.y);
    }
    if (!cell.empty())
      p.polygon(x, y, {129, 140, 248}, false);
  }
  std::vector<double> x, y;
  for (auto q : r.sites) {
    x.push_back(q.x);
    y.push_back(q.y);
  }
  p.points(x, y, "actual sites", {220, 38, 38}, 6);
  double span = std::max(r.box.xmax - r.box.xmin, r.box.ymax - r.box.ymin),
         cx = (r.box.xmin + r.box.xmax) / 2, cy = (r.box.ymin + r.box.ymax) / 2;
  p.size(760, 800)
      .margins(80, 40, 72, 88)
      .x_limits(cx - span / 2, cx + span / 2)
      .y_limits(cy - span / 2, cy + span / 2)
      .title("Actual clipped nearest-site cells")
      .x_label("x")
      .y_label("y");
  return p;
}
/// @brief Generate the original report's24 independent seeded sites.
std::vector<Point2D> voronoi_example_sites() {
  std::mt19937                           rng(1234567);
  std::uniform_real_distribution<double> u(1, 9);
  std::vector<Point2D>                   out;
  for (int i = 0; i < 24; ++i)
    out.push_back({u(rng), u(rng)});
  return out;
}
} // namespace datamunge::geometry
