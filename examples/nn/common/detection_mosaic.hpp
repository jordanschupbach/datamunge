// Four-image mosaic and horizontal flip for grayscale detection scenes (port
// of tflow's common/data/detection_mosaic.h).
//
// Each selected image is bilinearly reduced by two and placed in one
// quadrant of a same-size canvas. At this exact scale the pixel-centre
// bilinear samples are the averages of 2x2 source pixels. Boxes are scaled
// and translated with their images; labels stay attached to their boxes.
// This fixed-centre mosaic is a small-image adaptation of the random-centre
// crop mosaics used by YOLO training. Only the caller's selected source rows
// are used, so training can exclude validation and test examples explicitly.
#pragma once

#include "common/detection_scenes.hpp"

namespace detector {

inline DetectionScenes MosaicScenes(const DetectionScenes& source, const std::vector<std::array<int, 4>>& selections) {
  if (source.height < 2 || source.width < 2 || source.height % 2 || source.width % 2)
    throw std::invalid_argument("mosaic requires positive even image dimensions");
  DetectionScenes result;
  result.height = source.height;
  result.width  = source.width;
  result.pixels.resize(selections.size() * std::size_t(source.height) * std::size_t(source.width));
  result.boxes.resize(selections.size());
  result.labels.resize(selections.size());
  const int h = source.height / 2, w = source.width / 2;
  for (std::size_t n = 0; n < selections.size(); ++n)
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
      const int row = selections[n][std::size_t(quadrant)];
      if (row < 0 || row >= source.size() || source.boxes[std::size_t(row)].size() != source.labels[std::size_t(row)].size())
        throw std::invalid_argument("invalid mosaic source row");
      const auto* image = source.image(row);
      const int   y0 = (quadrant / 2) * h, x0 = (quadrant % 2) * w;
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
          const int pixel = 2 * r * source.width + 2 * c;
          const int sum   = image[pixel] + image[pixel + 1] + image[pixel + source.width] + image[pixel + source.width + 1];
          result.pixels[(n * std::size_t(source.height) + std::size_t(y0 + r)) * std::size_t(source.width) + std::size_t(x0 + c)] =
              std::uint8_t((sum + 2) / 4);
        }
      for (std::size_t k = 0; k < source.boxes[std::size_t(row)].size(); ++k) {
        const auto& b = source.boxes[std::size_t(row)][k];
        result.boxes[n].push_back({y0 + .5f * b[0], x0 + .5f * b[1], y0 + .5f * b[2], x0 + .5f * b[3]});
        result.labels[n].push_back(source.labels[std::size_t(row)][k]);
      }
    }
  return result;
}

inline DetectionScenes CopyDetectionRows(const DetectionScenes& source, const std::vector<int>& rows) {
  DetectionScenes result;
  result.height = source.height;
  result.width  = source.width;
  const std::size_t pixels = std::size_t(source.height) * std::size_t(source.width);
  result.pixels.resize(rows.size() * pixels);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int row = rows[i];
    if (row < 0 || row >= source.size()) throw std::out_of_range("detection row");
    std::copy_n(source.image(row), pixels, result.pixels.data() + i * pixels);
    result.boxes.push_back(source.boxes[std::size_t(row)]);
    result.labels.push_back(source.labels[std::size_t(row)]);
  }
  return result;
}

inline void FlipDetectionScene(DetectionScenes& scenes, int row) {
  if (row < 0 || row >= scenes.size()) throw std::out_of_range("detection flip row");
  auto* image = scenes.pixels.data() + std::size_t(row) * std::size_t(scenes.height) * std::size_t(scenes.width);
  for (int r = 0; r < scenes.height; ++r)
    for (int c = 0; c < scenes.width / 2; ++c) std::swap(image[r * scenes.width + c], image[r * scenes.width + scenes.width - 1 - c]);
  for (auto& b : scenes.boxes[std::size_t(row)]) {
    const float left = float(scenes.width) - b[3];
    b[3]             = float(scenes.width) - b[1];
    b[1]             = left;
  }
}

}  // namespace detector
