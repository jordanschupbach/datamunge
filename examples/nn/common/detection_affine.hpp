// Uniform scale, translation and exposure for grayscale detection scenes
// (port of tflow's common/data/detection_affine.h).
//
// Coordinates describe pixel edges. Scaling is about the image centre, then
// shift_y/shift_x translate in pixels. Images use inverse pixel-centre bilinear
// sampling with zero padding; boxes use the same forward transform, are clipped
// to the canvas, and are removed only when fully outside. Labels stay attached.
// Parameters and source rows are explicit: the training caller owns randomness
// and can exclude validation/test rows. (tflow also rejects scenes carrying
// instance masks; detector::DetectionScenes has no masks.)
#pragma once

#include "common/detection_scenes.hpp"

namespace detector {

struct DetectionTransform {
  double scale = 1, shift_y = 0, shift_x = 0, exposure = 1;
};

inline DetectionScenes TransformDetectionScenes(const DetectionScenes& source, const std::vector<int>& rows,
                                                const std::vector<DetectionTransform>& transforms) {
  if (source.height < 1 || source.width < 1 || rows.size() != transforms.size() || source.labels.size() != source.boxes.size() ||
      source.pixels.size() != std::size_t(source.size()) * std::size_t(source.height) * std::size_t(source.width))
    throw std::invalid_argument("affine detection transform requires valid grayscale scenes and matching transforms");
  DetectionScenes result;
  result.height = source.height;
  result.width  = source.width;
  const std::size_t pixels = std::size_t(source.height) * std::size_t(source.width);
  result.pixels.resize(rows.size() * pixels);
  result.boxes.resize(rows.size());
  result.labels.resize(rows.size());
  for (std::size_t n = 0; n < rows.size(); ++n) {
    const int   row = rows[n];
    const auto& t   = transforms[n];
    if (row < 0 || row >= source.size() || source.boxes[std::size_t(row)].size() != source.labels[std::size_t(row)].size() ||
        !(t.scale > 0) || !(t.exposure > 0) || !std::isfinite(t.scale) || !std::isfinite(t.exposure) || !std::isfinite(t.shift_y) ||
        !std::isfinite(t.shift_x))
      throw std::invalid_argument("invalid affine detection row or transform");
    const double offset_y = source.height * .5 * (1 - t.scale) + t.shift_y;
    const double offset_x = source.width * .5 * (1 - t.scale) + t.shift_x;
    if (!std::isfinite(offset_y) || !std::isfinite(offset_x)) throw std::invalid_argument("affine detection offsets overflow");
    const auto* input = source.image(row);
    for (int y = 0; y < source.height; ++y)
      for (int x = 0; x < source.width; ++x) {
        const double sy = (y + .5 - offset_y) / t.scale - .5, sx = (x + .5 - offset_x) / t.scale - .5;
        double       value = 0;
        if (sy > -1 && sy < source.height && sx > -1 && sx < source.width) {
          const int    y0 = int(std::floor(sy)), x0 = int(std::floor(sx));
          const double fy = sy - y0, fx = sx - x0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
              const int iy = y0 + dy, ix = x0 + dx;
              if (iy >= 0 && iy < source.height && ix >= 0 && ix < source.width)
                value += input[std::size_t(iy) * std::size_t(source.width) + std::size_t(ix)] * (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx);
            }
        }
        result.pixels[n * pixels + std::size_t(y) * std::size_t(source.width) + std::size_t(x)] =
            std::uint8_t(std::lround(std::clamp(value * t.exposure, 0., 255.)));
      }
    for (std::size_t k = 0; k < source.boxes[std::size_t(row)].size(); ++k) {
      const auto& b = source.boxes[std::size_t(row)][k];
      for (float coordinate : b)
        if (!std::isfinite(coordinate)) throw std::invalid_argument("nonfinite source box");
      if (!(b[0] < b[2] && b[1] < b[3]) || b[0] < 0 || b[1] < 0 || b[2] > source.height || b[3] > source.width)
        throw std::invalid_argument("source box must have positive area within its image");
      const Box transformed = {float(std::clamp(std::fma(t.scale, double(b[0]), offset_y), 0., double(source.height))),
                               float(std::clamp(std::fma(t.scale, double(b[1]), offset_x), 0., double(source.width))),
                               float(std::clamp(std::fma(t.scale, double(b[2]), offset_y), 0., double(source.height))),
                               float(std::clamp(std::fma(t.scale, double(b[3]), offset_x), 0., double(source.width)))};
      if (BoxArea(transformed) > 0) {
        result.boxes[n].push_back(transformed);
        result.labels[n].push_back(source.labels[std::size_t(row)][k]);
      }
    }
  }
  return result;
}

}  // namespace detector
