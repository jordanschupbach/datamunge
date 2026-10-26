// Class-specific instance-mask utilities: support IoU, pasted masks and
// scored mask AP (port of tflow's common/ml/instance_segmentation.h), plus
// digit scenes that keep each object's binary support (tflow's
// SceneConfig::retain_instance_masks).
//
// InstanceScenes extends detector::DetectionScenes with per-object masks in
// scene pixel coordinates (pixels of the rescaled digit brighter than 32).
// MakeInstanceScenes repeats detector::MakeDetectionScenes draw for draw
// (same std::mt19937_64 stream as tflow), so the images, boxes and labels are
// identical to MakeDetectionScenes with the same configuration; it is a copy
// rather than a change to the shared generator.
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

namespace instance {

namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;
using detector::DetectionScenes;

using BinaryMask = std::vector<std::uint8_t>;

struct InstanceScenes : DetectionScenes {
  std::vector<std::vector<BinaryMask>> instance_masks;  // [scene][object] -> H x W support
};

inline InstanceScenes MakeInstanceScenes(const detector::Digits& digits, const detector::SceneConfig& cfg) {
  if (digits.size() < 1 || cfg.max_objects < 1 || cfg.min_size < 4 || cfg.max_size < cfg.min_size ||
      cfg.max_size > std::min(cfg.height, cfg.width))
    throw std::invalid_argument("invalid detection scene configuration");
  InstanceScenes out;
  out.height = cfg.height;
  out.width  = cfg.width;
  out.pixels.assign(std::size_t(cfg.scenes) * cfg.height * cfg.width, 0);
  std::mt19937_64                    random(cfg.seed);
  std::uniform_int_distribution<int> count(1, cfg.max_objects), pick(0, digits.size() - 1), size(cfg.min_size, cfg.max_size);
  for (int n = 0; n < cfg.scenes; ++n) {
    std::vector<Box>        boxes;
    std::vector<int>        classes;
    std::vector<BinaryMask> masks;
    std::uint8_t*           canvas  = out.pixels.data() + std::size_t(n) * cfg.height * cfg.width;
    const int               objects = count(random);
    for (int k = 0; k < objects; ++k) {
      for (int attempt = 0; attempt < 30; ++attempt) {
        const int                          digit = pick(random), cell = size(random);
        std::uniform_int_distribution<int> top(0, cfg.height - cell), left(0, cfg.width - cell);
        const int                          y0 = top(random), x0 = left(random);
        const std::uint8_t*                source = digits.pixels.data() + std::size_t(digit) * 784;
        std::vector<std::uint8_t>          patch(std::size_t(cell) * cell);
        int                                ymin = cell, xmin = cell, ymax = -1, xmax = -1;
        for (int r = 0; r < cell; ++r)
          for (int c = 0; c < cell; ++c) {
            const double sy = (r + .5) * 28. / cell - .5, sx = (c + .5) * 28. / cell - .5;
            const int    iy = int(std::floor(sy)), ix = int(std::floor(sx));
            double       v  = 0;
            for (int a = 0; a <= 1; ++a)
              for (int b = 0; b <= 1; ++b) {
                const int yy = std::clamp(iy + a, 0, 27), xx = std::clamp(ix + b, 0, 27);
                v += (a ? sy - iy : 1 - (sy - iy)) * (b ? sx - ix : 1 - (sx - ix)) * source[yy * 28 + xx];
              }
            auto& p = patch[std::size_t(r) * cell + c];
            p       = std::uint8_t(std::clamp(std::lround(v), 0L, 255L));
            if (p > 32) {
              ymin = std::min(ymin, r);
              xmin = std::min(xmin, c);
              ymax = std::max(ymax, r);
              xmax = std::max(xmax, c);
            }
          }
        if (ymax < 0) continue;
        const Box box   = {float(y0 + ymin), float(x0 + xmin), float(y0 + ymax + 1), float(x0 + xmax + 1)};
        bool      clear = true;
        for (const auto& b : boxes)
          if (detector::BoxIoU(b, box) > cfg.max_overlap) {
            clear = false;
            break;
          }
        if (!clear) continue;
        BinaryMask mask(std::size_t(cfg.height) * cfg.width, 0);
        for (int r = 0; r < cell; ++r)
          for (int c = 0; c < cell; ++c) {
            std::uint8_t& p = canvas[(y0 + r) * cfg.width + x0 + c];
            p               = std::max(p, patch[std::size_t(r) * cell + c]);
            mask[std::size_t((y0 + r) * cfg.width + x0 + c)] = patch[std::size_t(r) * cell + c] > 32;
          }
        masks.push_back(std::move(mask));
        boxes.push_back(box);
        classes.push_back(digits.labels[std::size_t(digit)]);
        break;
      }
    }
    out.boxes.push_back(boxes);
    out.labels.push_back(classes);
    out.instance_masks.push_back(std::move(masks));
  }
  return out;
}

struct MaskDetection {
  Detection  box;
  BinaryMask mask;
};

inline double MaskIoU(const BinaryMask& a, const BinaryMask& b) {
  if (a.size() != b.size()) throw std::invalid_argument("mask grids differ");
  int intersection = 0, united = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    intersection += (a[i] != 0) && (b[i] != 0);
    united += (a[i] != 0) || (b[i] != 0);
  }
  return united ? double(intersection) / united : 0;
}

/// Bilinear paste of class `label` of a [side, side, classes] probability map into the box, threshold .5.
inline BinaryMask PasteMask(const float* probabilities, int side, int classes, int label, const Box& box, int height,
                            int width) {
  BinaryMask mask(std::size_t(height) * width, 0);
  if (label < 0 || label >= classes || side < 1 || !(box[2] > box[0] && box[3] > box[1]))
    throw std::invalid_argument("invalid instance mask geometry/class");
  for (int y = std::max(0, int(std::floor(box[0]))); y < std::min(height, int(std::ceil(box[2]))); ++y)
    for (int x = std::max(0, int(std::floor(box[1]))); x < std::min(width, int(std::ceil(box[3]))); ++x) {
      if (y + .5 < box[0] || y + .5 >= box[2] || x + .5 < box[1] || x + .5 >= box[3]) continue;
      const float sy = std::clamp((y + .5f - box[0]) / (box[2] - box[0]) * side - .5f, 0.f, float(side - 1)),
                  sx = std::clamp((x + .5f - box[1]) / (box[3] - box[1]) * side - .5f, 0.f, float(side - 1));
      const int   y0 = int(sy), x0 = int(sx);
      float       value = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
          value += (dy ? sy - y0 : 1 - (sy - y0)) * (dx ? sx - x0 : 1 - (sx - x0)) *
                   probabilities[(std::min(y0 + dy, side - 1) * side + std::min(x0 + dx, side - 1)) * classes + label];
      mask[std::size_t(y) * width + std::size_t(x)] = value >= .5f;
    }
  return mask;
}

/// All-point AP of scored masks against the scenes' instance supports, averaged over classes with targets.
inline double MaskAveragePrecision(const std::vector<MaskDetection>& detections, const InstanceScenes& scenes,
                                   int classes, double threshold) {
  if (scenes.instance_masks.size() != std::size_t(scenes.size())) throw std::invalid_argument("instance support missing");
  double sum     = 0;
  int    present = 0;
  for (int c = 0; c < classes; ++c) {
    int positives = 0;
    for (const auto& labels : scenes.labels) positives += int(std::count(labels.begin(), labels.end(), c));
    if (!positives) continue;
    ++present;
    std::vector<const MaskDetection*> sorted;
    for (const auto& d : detections)
      if (d.box.label == c) sorted.push_back(&d);
    std::stable_sort(sorted.begin(), sorted.end(), [](auto a, auto b) { return a->box.score > b->box.score; });
    std::vector<std::vector<bool>> used;
    for (const auto& masks : scenes.instance_masks) used.emplace_back(masks.size(), false);
    std::vector<double> precision, recall;
    int                 tp = 0, fp = 0;
    for (auto d : sorted) {
      const int row = d->box.image;
      if (row < 0 || row >= scenes.size()) throw std::out_of_range("mask detection image");
      int    best    = -1;
      double overlap = -1;
      for (std::size_t k = 0; k < scenes.labels[std::size_t(row)].size(); ++k)
        if (scenes.labels[std::size_t(row)][k] == c) {
          const double iou = MaskIoU(d->mask, scenes.instance_masks[std::size_t(row)][k]);
          if (iou > overlap) {
            overlap = iou;
            best    = int(k);
          }
        }
      if (best >= 0 && overlap >= threshold && !used[std::size_t(row)][std::size_t(best)]) {
        used[std::size_t(row)][std::size_t(best)] = true;
        ++tp;
      } else
        ++fp;
      precision.push_back(double(tp) / (tp + fp));
      recall.push_back(double(tp) / positives);
    }
    for (int i = int(precision.size()) - 2; i >= 0; --i)
      precision[std::size_t(i)] = std::max(precision[std::size_t(i)], precision[std::size_t(i) + 1]);
    double last = 0, ap = 0;
    for (std::size_t k = 0; k < precision.size(); ++k) {
      ap += (recall[k] - last) * precision[k];
      last = recall[k];
    }
    sum += ap;
  }
  return present ? sum / present : 0;
}

/// First four scenes: image | true class masks | predicted class masks (higher scores painted last).
inline void ExportInstanceMasks(const InstanceScenes& scenes, const std::vector<MaskDetection>& detections,
                                const std::filesystem::path& filename) {
  const int          count = std::min(4, scenes.size()), h = scenes.height, w = scenes.width;
  std::vector<float> rgb(std::size_t(count) * h * 3 * w * 3, 0.f);
  auto paint = [&](int row, int col, const BinaryMask& mask, int label) {
    check::Require(mask.size() == std::size_t(h * w), "preview instance mask dimensions");
    const float color[] = {float(80 + (37 * label) % 176), float(80 + (83 * label + 90) % 176),
                           float(80 + (113 * label + 30) % 176)};
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (mask[std::size_t(y * w + x)])
          for (int k = 0; k < 3; ++k) rgb[((std::size_t(row) * h + y) * 3 * w + std::size_t(col * w + x)) * 3 + k] = color[k];
  };
  for (int row = 0; row < count; ++row) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int k = 0; k < 3; ++k)
          rgb[((std::size_t(row) * h + y) * 3 * w + std::size_t(x)) * 3 + k] = scenes.image(row)[y * w + x];
    for (std::size_t k = 0; k < scenes.instance_masks[std::size_t(row)].size(); ++k)
      paint(row, 1, scenes.instance_masks[std::size_t(row)][k], scenes.labels[std::size_t(row)][k]);
  }
  std::vector<const MaskDetection*> ordered;
  for (const auto& d : detections)
    if (d.box.image < count) ordered.push_back(&d);
  std::stable_sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return a->box.score < b->box.score; });
  for (auto d : ordered) paint(d->box.image, 2, d->mask, d->box.label);
  nn::save_png(nn::from_vector(rgb, {count * h, 3 * w, 3}), filename.string(), 0, 255);
  std::cout << "  image/truth/predicted class-mask preview: " << filename.string() << std::endl;
}

}  // namespace instance
