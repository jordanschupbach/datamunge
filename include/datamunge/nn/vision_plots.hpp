#pragma once
/// @file vision_plots.hpp
/// @brief Tensor heatmaps and image contact sheets as ordinary local SVG figures.
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/plot/plot.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::nn {
/// @brief Plot a floating matrix as an image with the first row at the top.
///
/// Presentation only: detaches values to host coordinates, retaining no graph.
/// Cells use RPlot::image; no image XML is printed. All values must be finite.
/// Constant matrices get a one-unit colour range. O(HW) figure primitives.
/// @param matrix Nonempty numeric [H,W], both dimensions at least two.
/// @param low Lower colour limit; NaN selects the data minimum.
/// @param high Upper colour limit; NaN selects the data maximum.
/// @return Editable RPlot; caller sets title/size and saves to a local file.
/// @throws std::invalid_argument For incompatible shape or invalid limits.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto plot=n::plot_tensor_map(n::tensor({0,1,2,3},{2,2}));
/// @endcode
/// @see graphics::RPlot::image(), plot_tensor_images()
inline graphics::RPlot
plot_tensor_map(const Tensor &matrix, double low = std::numeric_limits<double>::quiet_NaN(),
                double high = std::numeric_limits<double>::quiet_NaN()) {
  if (!matrix.defined() || matrix.dim() != 2 || matrix.size(0) < 2 || matrix.size(1) < 2)
    throw std::invalid_argument("tensor map requires [H,W] with both axes at least two");
  const auto h = matrix.size(0), w = matrix.size(1);
  const auto values = matrix.to_vector<double>();
  for (auto v : values)
    if (!std::isfinite(v))
      throw std::invalid_argument("finite map values required");
  if (std::isnan(low))
    low = *std::min_element(values.begin(), values.end());
  if (std::isnan(high))
    high = *std::max_element(values.begin(), values.end());
  if (high == low)
    high = low + 1;
  if (!std::isfinite(low) || !std::isfinite(high) || high <= low)
    throw std::invalid_argument("invalid colour limits");
  std::vector<double> x(w), y(h);
  std::iota(x.begin(), x.end(), 0.);
  std::iota(y.begin(), y.end(), 0.);
  std::vector<std::vector<double>> z(w, std::vector<double>(h));
  for (std::int64_t a = 0; a < w; ++a)
    for (std::int64_t b = 0; b < h; ++b)
      z[a][b] = values[(h - 1 - b) * w + a];
  auto plot = graphics::RPlot::create();
  plot.image(x, y, z, graphics::terrain_colors(32), low, high)
      .x_label("column")
      .y_label("row from bottom");
  return plot;
}

/// @brief Preview grayscale or RGB tensor images with no embedded raster payload.
///
/// Tiles [N,H,W,C] images with image_grid and draws RGB pixel rectangles into
/// an editable SVG plot. Conversion uses to_image clipping/rounding; it is a
/// presentation operation with no gradient. Use small batches/thumbnails because
/// SVG storage and work are O(NHW). The first image row appears at the top.
/// @param batch Nonempty [N,H,W,1] or [N,H,W,3], numeric pixel intensities.
/// @param columns Positive tiles per row, default five.
/// @param low Intensity mapped to black, default zero.
/// @param high Intensity mapped to white, default one; must exceed low.
/// @return Editable RPlot; set title/size and save to build/ before export.
/// @throws std::invalid_argument For incompatible shapes, columns or limits.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto plot=n::plot_tensor_images(n::zeros({2,4,4,3}),2);
/// @endcode
/// @see image_grid(), to_image(), plot_tensor_map()
inline graphics::RPlot plot_tensor_images(const Tensor &batch, std::int64_t columns = 5,
                                          double low = 0, double high = 1) {
  if (!batch.defined() || batch.dim() != 4 || batch.numel() < 1 ||
      (batch.size(3) != 1 && batch.size(3) != 3) || columns < 1 || !std::isfinite(low) ||
      !std::isfinite(high) || high <= low)
    throw std::invalid_argument(
        "image preview requires [N,H,W,1 or 3], positive columns and valid limits");
  const auto image = to_image(image_grid(batch, columns, 2, low), low, high);
  auto plot = graphics::RPlot::create();
  const auto &bytes = image.data();
  for (int y = 0; y < image.height(); ++y)
    for (int x = 0; x < image.width(); ++x) {
      const auto offset = (y * image.width() + x) * image.channels();
      const int red = bytes[offset], green = image.channels() == 1 ? red : bytes[offset + 1],
                blue = image.channels() == 1 ? red : bytes[offset + 2];
      const double row = image.height() - 1 - y;
      plot.polygon({x - .5, x + .5, x + .5, x - .5}, {row - .5, row - .5, row + .5, row + .5},
                   {red, green, blue});
    }
  return plot.x_label("pixel column").y_label("pixel row from bottom");
}
} // namespace datamunge::nn
