// Small panoptic teaching data: MNIST instances over two textured materials
// (port of tflow's common/ml/panoptic_scenes.h). Retain original instance
// supports before filling non-instance pixels with background texture.
// Semantic 0 = all things, 1/2 = stuff categories 10/11 (split at a
// per-scene row). Instance intensities are restored only at the original
// support pixels; labels/masks come from the original digit scene, never from
// a threshold of the textured input.
#pragma once

#include <algorithm>
#include <cstdint>
#include <datamunge/nn/image_io.hpp>
#include <filesystem>
#include <map>
#include <vector>

#include "common/checks.hpp"
#include "common/instance_segmentation.hpp"
#include "common/panoptic.hpp"

namespace panoptic {

namespace nn = datamunge::nn;

struct PanopticScenes {
  instance::InstanceScenes      instances;
  std::vector<std::vector<int>> semantic;
  std::vector<PanopticMap>      panoptic;
};

inline PanopticScenes MakePanopticScenes(const detector::Digits& digits, int count, int seed) {
  detector::SceneConfig cfg;
  cfg.scenes      = count;
  cfg.seed        = std::uint64_t(seed);
  cfg.max_overlap = 0;
  PanopticScenes out;
  out.instances              = instance::MakeInstanceScenes(digits, cfg);
  const auto original_pixels = out.instances.pixels;
  const int  h = out.instances.height, w = out.instances.width;
  out.semantic.resize(std::size_t(count));
  out.panoptic.resize(std::size_t(count));
  for (int row = 0; row < count; ++row) {
    const int objects = int(out.instances.boxes[std::size_t(row)].size()), split = h / 2 + (seed + 7 * row) % 9 - 4;
    auto&     semantic = out.semantic[std::size_t(row)];
    auto&     map      = out.panoptic[std::size_t(row)];
    semantic.resize(std::size_t(h * w));
    map.ids.resize(std::size_t(h * w));
    map.segments[objects + 1] = {10, false, false};
    map.segments[objects + 2] = {11, false, false};
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int label = y < split ? 1 : 2, pixel = y * w + x;
        semantic[std::size_t(pixel)] = label;
        map.ids[std::size_t(pixel)]  = objects + label;
        out.instances.pixels[std::size_t(row * h * w + pixel)] =
            std::uint8_t((label == 1 ? 40 : 100) + (13 * y + 7 * x + 3 * row) % 11);
      }
  }
  for (int row = 0; row < count; ++row) {
    auto&     map     = out.panoptic[std::size_t(row)];
    const int objects = int(out.instances.boxes[std::size_t(row)].size());
    for (int k = 0; k < objects; ++k) {
      map.segments[k + 1] = {out.instances.labels[std::size_t(row)][std::size_t(k)], true, false};
      for (int pixel = 0; pixel < h * w; ++pixel)
        if (out.instances.instance_masks[std::size_t(row)][std::size_t(k)][std::size_t(pixel)]) {
          out.instances.pixels[std::size_t(row * h * w + pixel)] = original_pixels[std::size_t(row * h * w + pixel)];
          out.semantic[std::size_t(row)][std::size_t(pixel)]     = 0;
          map.ids[std::size_t(pixel)]                            = k + 1;
        }
    }
    std::map<int, int> areas;
    for (int id : map.ids) ++areas[id];
    for (auto it = map.segments.begin(); it != map.segments.end();)
      if (!areas[it->first])
        it = map.segments.erase(it);
      else
        ++it;
    PanopticAreas(map);
  }
  return out;
}

/// Semantic targets [rows, H, W] (int32).
inline nn::Tensor PanopticSemanticTargets(const PanopticScenes& scenes, const std::vector<int>& rows) {
  const std::size_t         pixels = std::size_t(scenes.instances.height) * scenes.instances.width;
  std::vector<std::int32_t> targets(rows.size() * pixels);
  for (std::size_t i = 0; i < rows.size(); ++i)
    std::copy_n(scenes.semantic.at(std::size_t(rows[i])).data(), pixels, targets.data() + i * pixels);
  return nn::from_vector(targets, {std::int64_t(rows.size()), scenes.instances.height, scenes.instances.width});
}

/// First four scenes: input | predicted semantic | true panoptic | predicted panoptic.
inline void PanopticPreview(const PanopticScenes& data, const std::vector<std::vector<int>>& semantic,
                            const std::vector<PanopticMap>& predicted, const std::filesystem::path& filename) {
  const int count = std::min(4, data.instances.size()), h = data.instances.height, w = data.instances.width;
  check::Require(semantic.size() == std::size_t(data.instances.size()) && predicted.size() == semantic.size(),
                 "panoptic preview covers every test image");
  std::vector<float> rgb(std::size_t(count) * h * 4 * w * 3, 0.f);
  auto at = [&](int row, int pixel, int column, int c) {
    return ((std::size_t(row) * h + std::size_t(pixel / w)) * 4 * w + std::size_t(column * w + pixel % w)) * 3 + std::size_t(c);
  };
  auto paint = [&](int row, int pixel, int column, int label) {
    const float color[] = {float(label == 10 ? 80 : label == 11 ? 170 : 80 + (37 * label) % 176),
                           float(label == 10 ? 80 : label == 11 ? 160 : 80 + (83 * label + 90) % 176),
                           float(label == 10 ? 120 : label == 11 ? 100 : 80 + (113 * label + 30) % 176)};
    for (int c = 0; c < 3; ++c) rgb[at(row, pixel, column, c)] = color[c];
  };
  for (int row = 0; row < count; ++row)
    for (int pixel = 0; pixel < h * w; ++pixel) {
      for (int c = 0; c < 3; ++c) rgb[at(row, pixel, 0, c)] = data.instances.image(row)[pixel];
      if (semantic[std::size_t(row)][std::size_t(pixel)]) paint(row, pixel, 1, semantic[std::size_t(row)][std::size_t(pixel)] + 9);
      const int gt = data.panoptic[std::size_t(row)].ids[std::size_t(pixel)],
                found = predicted[std::size_t(row)].ids[std::size_t(pixel)];
      if (gt) paint(row, pixel, 2, data.panoptic[std::size_t(row)].segments.at(gt).label);
      if (found) paint(row, pixel, 3, predicted[std::size_t(row)].segments.at(found).label);
    }
  nn::save_png(nn::from_vector(rgb, {count * h, 4 * w, 3}), filename.string(), 0, 255);
}

}  // namespace panoptic
