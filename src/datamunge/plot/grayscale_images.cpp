/// @file grayscale_images.cpp
/// @brief Fully validated grayscale pixel panels using native owned primitives.
#include <algorithm>
#include <cmath>
#include <datamunge/plot/grayscale_images.hpp>
#include <format>
#include <limits>
#include <stdexcept>
namespace datamunge::graphics {
Diagram diagram_grayscale_images(
    const std::vector<GrayscaleImagePanel>& panels,
    bool                                    show_values,
    std::size_t                             cap) {
  if (panels.empty() || panels.size() > 2)
    throw std::invalid_argument("gray images: require one or two panels");
  std::size_t total = 0, digits = 1;
  for (const auto& panel : panels) {
    const auto cols = panel.values.empty() ? 0 : panel.values[0].size();
    for (const auto& row : panel.values) {
      if (row.size() != cols)
        throw std::invalid_argument("gray images: ragged rows");
      for (double value : row) {
        if (!std::isfinite(value) || value < 0 || value > 255)
          throw std::invalid_argument("gray images: finite0..255 required");
        if (show_values)
          digits = std::max(digits, std::format("{:.4g}", value).size());
      }
    }
    if (cols && panel.values.size() > (std::numeric_limits<std::size_t>::max() - total) / cols)
      throw std::length_error("gray images: cell sum overflow");
    total += panel.values.size() * cols;
  }
  if (cap && total > cap)
    throw std::length_error("gray images: complete cell cap");
  Diagram       d(show_values ? std::max(52., 14. * digits * .7 + 16) : 10., 1100);
  DiagramStroke stroke;
  stroke.color  = {148, 163, 184, show_values ? 1. : 0.};
  stroke.width  = .7;
  double offset = 0;
  for (const auto& panel : panels) {
    const auto cols = panel.values.empty() ? 0 : panel.values[0].size();
    d.text({offset + (cols ? double(cols - 1) / 2 : 0), show_values ? 1.25 : 3.0}, panel.title, 20);
    for (std::size_t r = 0; r < panel.values.size(); ++r)
      for (std::size_t c = 0; c < cols; ++c) {
        const auto   value = panel.values[r][c];
        const int    gray  = static_cast<int>(std::lround(value));
        const double x = offset + c, y = -double(r);
        d.polygon(
            {{x - .5, y - .5}, {x + .5, y - .5}, {x + .5, y + .5}, {x - .5, y + .5}},
            RGB{gray, gray, gray},
            stroke);
        if (show_values)
          d.text(
              {x, y},
              std::format("{:.4g}", value),
              14,
              gray < 128 ? RGB{255, 255, 255} : RGB{15, 23, 42});
      }
    if (cols && !panel.values.empty()) {
      /// @brief Thin panel boundary keeps an all-white image's extent visible.
      DiagramStroke frame;
      frame.color       = {148, 163, 184};
      frame.width       = .7;
      const double left = offset - .5, right = offset + double(cols) - .5, top = .5,
                   bottom = .5 - double(panel.values.size());
      d.path({{left, top}, {right, top}, {right, bottom}, {left, bottom}, {left, top}}, frame);
    }
    offset += std::max(1., double(cols)) + 3;
  }
  return d;
}
} // namespace datamunge::graphics
