/// @file overlay.cpp
/// @brief Dependency-free RGBA text overlays using the shared native bitmap font.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/cad/render.hpp>
#include <stdexcept>

#include "../game/font.hpp"
namespace datamunge::cad {
void draw_text(
    RenderResult&      image,
    const std::string& text,
    int                x,
    int                y,
    int                scale,
    RgbaColorF         color) {
  if (image.width <= 0 || image.height <= 0 ||
      image.rgba.size() != std::size_t(image.width) * image.height * 4 || scale < 1 || scale > 16)
    throw std::invalid_argument("invalid text overlay image or scale");
  for (float v : {color.r, color.g, color.b, color.a})
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw std::invalid_argument("invalid text overlay color");
  std::int64_t pen_x = x, pen_y = y;
  for (unsigned char c : text) {
    if (c == '\n') {
      pen_x = x;
      pen_y += 8 * scale;
      continue;
    }
    const auto& glyph = game::detail::glyph(c);
    for (int row = 0; row < 7; ++row)
      for (int column = 0; column < 5; ++column)
        if (glyph[row] & (1u << (4 - column)))
          for (int dy = 0; dy < scale; ++dy)
            for (int dx = 0; dx < scale; ++dx) {
              auto px = pen_x + column * scale + dx, py = pen_y + row * scale + dy;
              if (px < 0 || py < 0 || px >= image.width || py >= image.height)
                continue;
              auto        i     = 4 * (std::size_t(py) * image.width + px);
              float       alpha = color.a + image.rgba[i + 3] / 255.f * (1 - color.a);
              const float channels[]{color.r, color.g, color.b};
              for (int k = 0; k < 3; ++k)
                image.rgba[i + k] =
                    alpha > 0 ? static_cast<unsigned char>(std::lround(
                                    255 *
                                    (channels[k] * color.a + image.rgba[i + k] / 255.f *
                                                                 (image.rgba[i + 3] / 255.f) *
                                                                 (1 - color.a)) /
                                    alpha))
                              : 0;
              image.rgba[i + 3] = static_cast<unsigned char>(std::lround(255 * alpha));
            }
    pen_x += 6 * scale;
  }
}
} // namespace datamunge::cad
