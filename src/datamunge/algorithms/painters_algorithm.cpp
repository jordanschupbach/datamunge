/// @file painters_algorithm.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/painters_algorithm.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
std::vector<int> painter_order(const std::vector<double>& depths) {
  if (depths.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("painter index range");
  for (double d : depths)
    if (!std::isfinite(d))
      throw std::invalid_argument("painter nonfinite depth");
  std::vector<int> order(depths.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return depths[a] > depths[b]; });
  return order;
}
/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {
/// @brief Shared validated renderer; optionally records each actual assignment.
PainterTrace
render(const std::vector<DepthRect>& rects, int width, int height, bool capture, std::size_t cap) {
  if (width < 0 || height < 0)
    throw std::invalid_argument("painter negative dimensions");
  std::vector<double> depths;
  depths.reserve(rects.size());
  for (const auto& r : rects)
    depths.push_back(r.depth);
  PainterTrace t;
  t.order = painter_order(depths);
  if (width && static_cast<std::size_t>(height) >
                   std::vector<int>().max_size() / static_cast<std::size_t>(width))
    throw std::length_error("painter raster size");
  t.raster.assign(height, std::vector<int>(width, 0));
  for (int idx : t.order) {
    const auto& r = rects[idx];
    for (int y = std::max(0, r.y0); y <= std::min(height - 1, r.y1); ++y)
      for (int x = std::max(0, r.x0); x <= std::min(width - 1, r.x1); ++x) {
        if (capture) {
          if (cap && t.writes.size() >= cap)
            throw std::length_error("painter write history cap");
          t.writes.push_back({static_cast<std::size_t>(idx), x, y, t.raster[y][x], r.color});
        }
        t.raster[y][x] = r.color;
      }
  }
  return t;
}
} // namespace
std::vector<std::vector<int>> painter_render(const std::vector<DepthRect>& r, int w, int h) {
  return render(r, w, h, false, 0).raster;
}
PainterTrace painter_trace(const std::vector<DepthRect>& r, int w, int h, std::size_t cap) {
  return render(r, w, h, true, cap);
}
} // namespace datamunge::algorithms
