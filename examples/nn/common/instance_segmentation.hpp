// Class-specific instance-mask utilities: support IoU, pasted masks and
// scored mask AP (port of tflow's common/ml/instance_segmentation.h), plus
// digit scenes that keep each object's binary support (tflow's
// SceneConfig::retain_instance_masks).
//
// InstanceScenes extends detector::DetectionScenes with per-object masks in
// scene pixel coordinates (pixels of the rescaled digit brighter than 32).
// Both instance and plain scenes use one canonical tensor sampler,with the
// same discrete mt19937_64 stream. Pixels,boxes and labels therefore match
// exactly between the two entry points. Tensor interpolation may differ by
// one byte from the original host implementation at borderline rounding.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/image_io.hpp>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/detection_scenes.hpp"
#include "datamunge/nn/instance_scene_models.hpp"

namespace instance {

namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;
using detector::DetectionScenes;

using BinaryMask = std::vector<std::uint8_t>;

struct InstanceScenes : DetectionScenes {
  std::vector<std::vector<BinaryMask>> instance_masks; // [scene][object] -> H x W support
};

inline InstanceScenes LegacyScenes(const nn::models::TensorInstanceScenes &data) {
  InstanceScenes out;
  static_cast<DetectionScenes &>(out) = detector::scene_adapter_detail::Legacy(data);
  for (const auto &row : data.masks) {
    std::vector<BinaryMask> masks;
    for (std::int64_t object = 0; object < row.shape()[0]; ++object) {
      const auto pixels = nn::slice_axis(row, 0, object, object + 1).to_vector<int>();
      masks.emplace_back(pixels.begin(), pixels.end());
    }
    out.instance_masks.push_back(std::move(masks));
  }
  return out;
}
inline InstanceScenes MakeInstanceScenes(const detector::Digits &digits,
                                         const detector::SceneConfig &cfg) {
  const auto input = detector::scene_adapter_detail::Inputs(digits);
  return LegacyScenes(nn::models::MakeTensorInstanceScenes(
      input.first, input.second, detector::scene_adapter_detail::Config(cfg)));
}

struct MaskDetection {
  Detection box;
  BinaryMask mask;
};

inline double MaskIoU(const BinaryMask &a, const BinaryMask &b) {
  return nn::models::InstanceMaskIoU(
             nn::not_equal(nn::from_vector(std::vector<int>(a.begin(), a.end())), 0.),
             nn::not_equal(nn::from_vector(std::vector<int>(b.begin(), b.end())), 0.))
      .item();
}

/// Bilinear paste of class `label` of a [side, side, classes] probability map into the box, threshold .5.
inline BinaryMask PasteMask(const float *probabilities, int side, int classes, int label,
                            const Box &box, int height, int width) {
  if (side < 1 || classes < 1)
    throw std::invalid_argument("invalid mask probability shape");
  const auto mask =
      nn::models::PasteInstanceMask(
          nn::from_vector(
              std::vector<float>(probabilities,
                                 probabilities + std::size_t(side) * side * classes),
              {side, side, classes}),
          label, nn::from_vector(std::vector<float>(box.begin(), box.end())), height, width)
          .to_vector<int>();
  return BinaryMask(mask.begin(), mask.end());
}

// Container ingress for annotation-only objectives/metrics;images carry geometry only.
inline nn::models::TensorInstanceScenes TensorAnnotations(const InstanceScenes &scenes) {
  if (scenes.instance_masks.size() != std::size_t(scenes.size()))
    throw std::invalid_argument("instance support missing");
  nn::models::TensorInstanceScenes data;
  data.images = nn::zeros({scenes.size(), scenes.height, scenes.width, 1});
  for (int row = 0; row < scenes.size(); ++row) {
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
    std::vector<int> masks;
    for (const auto &mask : scenes.instance_masks[row])
      masks.insert(masks.end(), mask.begin(), mask.end());
    data.masks.push_back(
        nn::not_equal(nn::from_vector(masks, {std::int64_t(scenes.instance_masks[row].size()),
                                              scenes.height, scenes.width}),
                      0.));
  }
  return data;
}

/// All-point AP of scored masks against the scenes' instance supports, averaged over classes with targets.
inline double MaskAveragePrecision(const std::vector<MaskDetection> &predictions,
                                   const InstanceScenes &scenes, int classes,
                                   double threshold) {
  const auto data = TensorAnnotations(scenes);
  std::vector<nn::Box> boxes;
  std::vector<float> scores;
  std::vector<std::int64_t> labels, images;
  std::vector<int> masks;
  for (const auto &d : predictions) {
    if (d.box.image < 0 || d.box.image >= scenes.size())
      throw std::out_of_range("mask detection image");
    boxes.push_back(d.box.box);
    scores.push_back(d.box.score);
    labels.push_back(d.box.label);
    images.push_back(d.box.image);
    masks.insert(masks.end(), d.mask.begin(), d.mask.end());
  }
  const auto M = std::int64_t(predictions.size());
  nn::models::TensorMaskDetections tensor{
      {nn::boxes_from_vector(boxes), nn::from_vector(scores), nn::from_vector(labels),
       nn::from_vector(images)},
      nn::not_equal(nn::from_vector(masks, {M, scenes.height, scenes.width}), 0.)};
  return nn::models::InstanceMaskAveragePrecision(tensor, data, classes, threshold).item();
}

/// First four scenes: image | true class masks | predicted class masks (higher scores painted last).
inline void ExportInstanceMasks(const InstanceScenes &scenes,
                                const std::vector<MaskDetection> &detections,
                                const std::filesystem::path &filename) {
  const int count = std::min(4, scenes.size()), h = scenes.height, w = scenes.width;
  std::vector<float> rgb(std::size_t(count) * h * 3 * w * 3, 0.f);
  auto paint = [&](int row, int col, const BinaryMask &mask, int label) {
    check::Require(mask.size() == std::size_t(h * w), "preview instance mask dimensions");
    const float color[] = {float(80 + (37 * label) % 176), float(80 + (83 * label + 90) % 176),
                           float(80 + (113 * label + 30) % 176)};
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (mask[std::size_t(y * w + x)])
          for (int k = 0; k < 3; ++k)
            rgb[((std::size_t(row) * h + y) * 3 * w + std::size_t(col * w + x)) * 3 + k] =
                color[k];
  };
  for (int row = 0; row < count; ++row) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int k = 0; k < 3; ++k)
          rgb[((std::size_t(row) * h + y) * 3 * w + std::size_t(x)) * 3 + k] =
              scenes.image(row)[y * w + x];
    for (std::size_t k = 0; k < scenes.instance_masks[std::size_t(row)].size(); ++k)
      paint(row, 1, scenes.instance_masks[std::size_t(row)][k],
            scenes.labels[std::size_t(row)][k]);
  }
  std::vector<const MaskDetection *> ordered;
  for (const auto &d : detections)
    if (d.box.image < count)
      ordered.push_back(&d);
  std::stable_sort(ordered.begin(), ordered.end(), [](auto a, auto b) {
    return a->box.score < b->box.score;
  });
  for (auto d : ordered)
    paint(d->box.image, 2, d->mask, d->box.label);
  nn::save_png(nn::from_vector(rgb, {count * h, 3 * w, 3}), filename.string(), 0, 255);
  std::cout << "  image/truth/predicted class-mask preview: " << filename.string()
            << std::endl;
}

} // namespace instance
