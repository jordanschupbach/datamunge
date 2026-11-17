#pragma once
#include "nn_interpretation_check_support.hpp"
namespace {
struct Canvas {
  int                       h, w;
  std::vector<std::uint8_t> px;
  Canvas(int height, int width, std::uint8_t fill = 220)
      : h(height), w(width), px(std::size_t(height) * width * 3, fill) {}
  std::uint8_t& at(int y, int x, int c) { return px[(std::size_t(y) * w + x) * 3 + c]; }
  void          Set(int y, int x, int c, double value) {
    at(y, x, c) = std::uint8_t(std::lround(std::clamp(value, 0., 255.)));
  }
  void Gray(int y, int x, double value) {
    for (int c = 0; c < 3; ++c)
      Set(y, x, c, value);
  }
};

// Small numeric labels;row-major channels are also documented in the gallery.
void Number(Canvas& canvas, int y, int x, int value) {
  const std::array<std::string, 10> font = {
      "111101101101111",
      "010110010010111",
      "111001111100111",
      "111001111001111",
      "101101111001001",
      "111100111001111",
      "111100111101111",
      "111001001001001",
      "111101111101111",
      "111101111001111"};
  for (char c : std::to_string(value)) {
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 3; ++b)
        if (font[c - '0'][a * 3 + b] == '1')
          for (int k = 0; k < 3; ++k)
            canvas.at(y + a, x + b, k) = 230;
    x += 4;
  }
}

std::uint8_t Byte(double value) {
  return std::uint8_t(std::lround(std::clamp(value, 0., 255.)));
}

/// features [N, H, W, C] (host values) -> all-channel mosaic of `image`.
Canvas Mosaic(
    const std::vector<float>& x,
    const nn::Shape&          shape,
    int                       image,
    bool                      shared,
    double                    maximum) {
  const int height = int(shape[1]), width = int(shape[2]), channels = int(shape[3]),
            tile = std::max(16, height), columns = channels == 64 ? 8 : 16,
            rows = (channels + columns - 1) / columns;
  auto at        = [&](int b, int y, int a, int c) {
    return double(x[((std::size_t(b) * height + y) * width + a) * channels + c]);
  };
  Canvas canvas(rows * (tile + 10) + 2, columns * (tile + 2) + 2, 24);
  for (int c = 0; c < channels; ++c) {
    double peak = 0;
    for (int y = 0; y < height; ++y)
      for (int a = 0; a < width; ++a)
        peak = std::max(peak, at(image, y, a, c));
    const double scale = shared ? maximum : peak;
    const int    oy = c / columns * (tile + 10) + 2, ox = c % columns * (tile + 2) + 2;
    Number(canvas, oy, ox, c);
    for (int y = 0; y < tile; ++y)
      for (int a = 0; a < tile; ++a) {
        const std::uint8_t v =
            scale > 0 ? Byte(255 * at(image, y * height / tile, a * width / tile, c) / scale) : 0;
        for (int k = 0; k < 3; ++k)
          canvas.at(oy + 8 + y, ox + a, k) = v;
      }
  }
  return canvas;
}

void RenderingChecks() {
  std::vector<float> zero(256, 0.f);
  auto               map = Mosaic(zero, {1, 2, 2, 64}, 0, false, 0);
  check::Require(map.at(10, 2, 0) == 0, "allzero channels render black,notNaN");
  std::vector<float> two(256, 2.f);
  map = Mosaic(two, {1, 2, 2, 64}, 0, false, 2);
  check::Require(map.at(10, 2, 0) == 255, "constantpositive ReLUchannel renderswhite,notzero");
  map = Mosaic(two, {1, 2, 2, 64}, 0, true, 4);
  check::Require(map.at(10, 2, 0) == 128, "shared scale retains relative response size");
  std::cout << "render scale fixtures:zero/constantpositive/sharedhalf values passed\n";
}

} // namespace
