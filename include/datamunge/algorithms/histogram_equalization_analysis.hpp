#pragma once
/// @file histogram_equalization_analysis.hpp
/// @brief Bounded original equalization fixture, exact16-bin summaries and native grayscale views.
#include <datamunge/algorithms/histogram_equalization.hpp>
#include <datamunge/plot/plot.hpp>
#include <string>
#include <cstdint>
namespace datamunge::algorithms {
/// @brief Generate the original low-contrast modular pattern with values96..159.
/// Value(i,j)=96+((7i+5j)mod64), computed with reduced indices to avoid overflow.
/// At64x64 each level appears64times. This is a deterministic synthetic fixture,
/// not a photograph or noise model. Geometry/caps precede all allocation.
/// @param rows Row count, default64, zero allowed.
/// @param columns Column count, default64, zero allowed; no-row output has no stored width.
/// @param options Complete row/pixel caps, default1m each; zero disables either.
/// @return Owned rectangular pattern in original row order.
/// @throws std::length_error On geometry/caps/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto x=dm::algorithms::equalization_pattern(2,2);
/// std::print("{} {}\n",x[0][0],x[1][1]); // 96 108
/// @endcode
/// @see histogram_equalize_trace()
inline EqualizationImage equalization_pattern(
    std::size_t         rows    = 64,
    std::size_t         columns = 64,
    EqualizationOptions options = {}) {
  (void)equalization_detail::dimensions(rows, columns, options);
  if (rows == 0) return {};
  EqualizationImage out(rows, std::vector<int>(columns));
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < columns; ++j)
      out[i][j] = 96 + int(((i % 64) * 7 + (j % 64) * 5) % 64);
  return out;
}
/// @brief Sum256 supplied level counts into16 exact contiguous16-level bins.
/// Bin0 covers0..15, bin15 covers240..255; no normalization or floating conversion.
/// Fixed work/storage, no allocation, retained references or shared state.
/// @param histogram Borrowed arbitrary size_t counts by level0..255.
/// @return Owned exact coarse counts, index0..15.
/// @throws std::length_error On any grouped count overflow.
/// @par Example
/// @code
/// const auto t=dm::algorithms::histogram_equalize_trace({{0,15,16,255}});
/// std::print("{}\n",dm::algorithms::equalization_hist16(t.histogram)[0]); // 2
/// @endcode
/// @see EqualizationTrace
inline std::array<std::size_t, 16> equalization_hist16(
    const std::array<std::size_t, 256>& histogram) {
  std::array<std::size_t, 16> out{};
  for (std::size_t i = 0; i < 256; ++i) {
    auto& n = out[i / 16];
    if (histogram[i] > std::numeric_limits<std::size_t>::max() - n)
      throw std::length_error("equalization grouped count overflow");
    n += histogram[i];
  }
  return out;
}
/// @brief Build a native grayscale grid view with a shared fixed0..255 palette scale.
/// Clamps pixel values for display and transposes row-major data to RPlot's
/// x-column/y-row convention. Original row0 appears at the top; axes are hidden.
/// Returns actual grid polygons. Data preparation and render storage are
/// O(rows+columns+pixels).
/// Image cells follow the plot panel geometry; callers choose layout dimensions.
/// No input references retained. Complete display cap precedes grid allocation.
/// @param image Borrowed rectangular image, both dimensions at least2.
/// @param title Owned display title, empty allowed.
/// @param max_pixels Complete display-cell cap, default16384; zero disables optional policy.
/// @return Owned native plot with one polygon series per pixel and fixed grayscale limits.
/// @throws std::invalid_argument On ragged shape or dimensions below2.
/// @throws std::length_error On geometry/display cap/container limits.
/// @throws std::bad_alloc On plot/grid allocation.
/// @par Example
/// @code
/// const auto p=dm::algorithms::equalization_image_plot({{0,255},{255,0}},"checker");
/// std::print("{}\n",p.series().size()); // 4
/// @endcode
/// @see graphics::RPlot::image(), equalization_pattern()
inline graphics::RPlot equalization_image_plot(
    const EqualizationImage& image,
    std::string              title,
    std::size_t              max_pixels = 16384) {
  (void)equalization_detail::geometry(image, {0, max_pixels});
  const auto rows = image.size(), columns = rows ? image.front().size() : 0;
  if (rows < 2 || columns < 2)
    throw std::invalid_argument("equalization image view needs at least2x2");
  std::vector<double> x(columns), y(rows);
  for (std::size_t j = 0; j < columns; ++j)
    x[j] = j;
  for (std::size_t i = 0; i < rows; ++i)
    y[i] = i;
  std::vector<std::vector<double>> z(columns, std::vector<double>(rows));
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < columns; ++j)
      z[j][rows - 1 - i] = std::clamp(image[i][j], 0, 255);
  std::vector<graphics::RGBA> palette;
  for (int i = 0; i < 256; ++i)
    palette.push_back({std::uint8_t(i), std::uint8_t(i), std::uint8_t(i), 255});
  auto p = graphics::RPlot::create();
  p.image(x, y, z, palette, 0, 255).title(title).hide_axes(true);
  return p;
}
} // namespace datamunge::algorithms
