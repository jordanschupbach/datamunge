// Synthetic detection scenes: MNIST digits scattered on a blank canvas (port
// of tflow's common/data/detection_scenes.h), plus the scene helpers of
// tflow's common/ml/detection.h that the detection ports share: batches as
// [-1, 1] image tensors, bilinear crops, squares around boxes and classical
// connected-component region proposals.
//
// Each scene places one to `max_objects` digits, each rescaled (bilinearly)
// so its 28x28 cell becomes `min_size`..`max_size` pixels, at a uniformly
// random position, rejecting placements whose box overlaps an earlier one
// with IoU above `max_overlap`. Overlapping strokes combine by maximum. The
// box of a digit is the tight box of its pixels brighter than 32 after
// rescaling, in pixel coordinates [ymin, xmin, ymax, xmax) and its class is
// the MNIST label. Scenes are generated deterministically from a seed with
// std::mt19937_64 and the same draw order as tflow, so a scene set is
// matched in discrete placement order. Tensor interpolation can differ by
// one byte at borderline rounding;training/test source splits remain distinct.
#pragma once
#include "common/dataset_assets.hpp"

#include <deque>
#include <random>

#include "common/detection.hpp"

namespace detector {

inline float BoxArea(const Box &b) {
  return std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
}
/// tflow's data::BoxIoU (zero for touching or disjoint boxes).
inline float BoxIoU(const Box &a, const Box &b) {
  const float h = std::min(a[2], b[2]) - std::max(a[0], b[0]),
              w = std::min(a[3], b[3]) - std::max(a[1], b[1]);
  if (h <= 0 || w <= 0)
    return 0;
  const float inter = h * w;
  return inter / (BoxArea(a) + BoxArea(b) - inter);
}

/// Raw MNIST digits: uint8 pixels [N, 28, 28] and labels.
struct Digits {
  std::vector<std::uint8_t> pixels;
  std::vector<int> labels;
  [[nodiscard]] int size() const {
    return int(labels.size());
  }
};
/// MNIST split ("train" or "t10k") from `directory` (IDX files, as tflow's data::LoadIdx).
inline Digits LoadDigits(const std::filesystem::path &directory, const std::string &split) {
  dataset_assets::EnsureMnist(directory.string());
  const auto locate = [&](const std::string &name) {
    const auto path = directory / name;
    return std::filesystem::exists(path) ? path.string() : path.string() + ".gz";
  };
  const auto images = locate(split + "-images-idx3-ubyte");
  const auto labels = locate(split + "-labels-idx1-ubyte");
  const auto x = nn::read_idx(images).to_vector<int>();
  const auto y = nn::read_idx(labels).to_vector<int>();
  check::Require(x.size() == y.size() * 784, "MNIST images are 28x28");
  Digits d;
  d.pixels.assign(x.begin(), x.end());
  d.labels = y;
  return d;
}

struct SceneConfig {
  int scenes = 1000, height = 64, width = 64, max_objects = 4, min_size = 16, max_size = 32;
  float max_overlap = .3f;
  std::uint64_t seed = 1;
};
/// @brief Internal ingress/egress adapters;all image/geometry arithmetic lives in tensor APIs.
namespace scene_adapter_detail {
/// @brief Convert raw byte/label containers to normalized tensor inputs without host image arithmetic.
/// @param digits Raw aligned28x28 byte images and integer labels.
/// @return Float32 normalized NHWC images and Int64 labels.
inline std::pair<nn::Tensor, nn::Tensor> Inputs(const Digits &digits) {
  return {
      nn::from_vector(std::vector<float>(digits.pixels.begin(), digits.pixels.end()),
                      {digits.size(), 28, 28, 1}) /
          255.,
      nn::from_vector(std::vector<std::int64_t>(digits.labels.begin(), digits.labels.end()))};
}
/// @brief Copy only scene dimensions,placement settings and RNG seed metadata.
/// @param cfg Legacy source-compatible settings.
/// @return Canonical tensor scene configuration.
inline nn::models::DetectionSceneConfig Config(const SceneConfig &cfg) {
  nn::models::DetectionSceneConfig out;
  out.scenes = cfg.scenes;
  out.height = cfg.height;
  out.width = cfg.width;
  out.max_objects = cfg.max_objects;
  out.min_size = cfg.min_size;
  out.max_size = cfg.max_size;
  out.max_overlap = cfg.max_overlap;
  out.seed = cfg.seed;
  return out;
}
/// @brief Present canonical tensor data using the legacy byte/box/label containers.
/// @param data Detached canonical scene data.
/// @return Independently owned legacy containers;quantization uses tensors.
inline DetectionScenes Legacy(const nn::models::TensorDetectionScenes &data) {
  DetectionScenes out;
  out.height = int(data.images.shape()[1]);
  out.width = int(data.images.shape()[2]);
  const auto bytes =
      nn::floor(data.images.to(nn::DType::Float64) * 255. + .5).to_vector<int>();
  out.pixels.assign(bytes.begin(), bytes.end());
  for (std::size_t row = 0; row < data.boxes.size(); ++row) {
    out.boxes.push_back(nn::boxes_to_vector(data.boxes[row]));
    out.labels.push_back(data.labels[row].to_vector<int>());
  }
  return out;
}
} // namespace scene_adapter_detail
/// @brief Forward scene generation to the canonical tensor sampler,retaining legacy container/API layout.
inline DetectionScenes MakeDetectionScenes(const Digits &digits, const SceneConfig &cfg) {
  const auto input = scene_adapter_detail::Inputs(digits);
  return scene_adapter_detail::Legacy(nn::models::MakeTensorDetectionScenes(
      input.first, input.second, scene_adapter_detail::Config(cfg)));
}

/// The standard train (6000 scenes, seed 11) and test (1000, seed 22) sets
/// of the digit-scene detection examples, at `size` pixels with digit cells
/// size/4..size/2.
inline std::pair<DetectionScenes, DetectionScenes>
StandardScenes(const std::filesystem::path &directory, int size = 64) {
  SceneConfig train_cfg, test_cfg;
  train_cfg.scenes = 6000;
  train_cfg.seed = 11;
  test_cfg.scenes = 1000;
  test_cfg.seed = 22;
  train_cfg.height = train_cfg.width = test_cfg.height = test_cfg.width = size;
  train_cfg.min_size = test_cfg.min_size = size / 4;
  train_cfg.max_size = test_cfg.max_size = size / 2;
  return {MakeDetectionScenes(LoadDigits(directory, "train"), train_cfg),
          MakeDetectionScenes(LoadDigits(directory, "t10k"), test_cfg)};
}

/// Scenes as a float batch [B, H, W, 1] scaled to [-1, 1].
inline nn::Tensor SceneImages(const DetectionScenes &scenes, const std::vector<int> &rows) {
  const std::int64_t pixels = std::int64_t(scenes.height) * scenes.width;
  nn::Tensor x = nn::empty({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  float *dst = x.data_ptr<float>();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::uint8_t *image = scenes.image(rows[i]);
    for (std::int64_t p = 0; p < pixels; ++p)
      dst[std::int64_t(i) * pixels + p] = image[p] / 127.5f - 1;
  }
  return x;
}

/// A box of one scene resampled to size x size (bilinear, zeros outside),
/// written as [-1, 1] floats to `out`.
inline void CropScene(const DetectionScenes &scenes, int row, const Box &box, int size,
                      float *out) {
  const std::uint8_t *image = scenes.image(row);
  const float h = box[2] - box[0], w = box[3] - box[1];
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c) {
      const float sy = box[0] + (r + .5f) * h / size - .5f,
                  sx = box[1] + (c + .5f) * w / size - .5f;
      const int y0 = int(std::floor(sy)), x0 = int(std::floor(sx));
      float v = 0;
      for (int a = 0; a <= 1; ++a)
        for (int b = 0; b <= 1; ++b) {
          const int yy = y0 + a, xx = x0 + b;
          if (yy < 0 || yy >= scenes.height || xx < 0 || xx >= scenes.width)
            continue;
          v += (a ? sy - y0 : 1 - (sy - y0)) * (b ? sx - x0 : 1 - (sx - x0)) *
               image[yy * scenes.width + xx];
        }
      out[r * size + c] = v / 127.5f - 1;
    }
}

/// A square around a box's centre, its side the longer box side times margin.
inline Box Square(const Box &b, float margin) {
  const float side = std::max(b[2] - b[0], b[3] - b[1]) * margin, cy = (b[0] + b[2]) / 2,
              cx = (b[1] + b[3]) / 2;
  return {cy - side / 2, cx - side / 2, cy + side / 2, cx + side / 2};
}

/// Classical region proposals: connected-component boxes of the image
/// thresholded at each level (8-connectivity, at least 9 pixels of box area)
/// and the merged boxes of component pairs lying within `gap` pixels; exact
/// duplicates dropped.
inline std::vector<Box> ComponentProposals(const DetectionScenes &scenes, int row,
                                           const std::vector<int> &thresholds, int gap) {
  const int h = scenes.height, w = scenes.width;
  const std::uint8_t *image = scenes.image(row);
  std::vector<Box> out;
  for (int threshold : thresholds) {
    std::vector<int> component(std::size_t(h) * w, -1);
    std::vector<Box> boxes;
    for (int start = 0; start < h * w; ++start) {
      if (image[start] <= threshold || component[std::size_t(start)] >= 0)
        continue;
      Box b = {float(h), float(w), 0, 0};
      std::deque<int> queue = {start};
      component[std::size_t(start)] = int(boxes.size());
      while (!queue.empty()) {
        const int p = queue.front();
        queue.pop_front();
        const int r = p / w, c = p % w;
        b = {std::min(b[0], float(r)), std::min(b[1], float(c)), std::max(b[2], float(r + 1)),
             std::max(b[3], float(c + 1))};
        for (int dr = -1; dr <= 1; ++dr)
          for (int dc = -1; dc <= 1; ++dc) {
            const int rr = r + dr, cc = c + dc;
            if (rr < 0 || rr >= h || cc < 0 || cc >= w || image[rr * w + cc] <= threshold ||
                component[std::size_t(rr * w + cc)] >= 0)
              continue;
            component[std::size_t(rr * w + cc)] = int(boxes.size());
            queue.push_back(rr * w + cc);
          }
      }
      if ((b[2] - b[0]) * (b[3] - b[1]) >= 9)
        boxes.push_back(b);
    }
    out.insert(out.end(), boxes.begin(), boxes.end());
    for (std::size_t i = 0; i < boxes.size(); ++i)
      for (std::size_t j = i + 1; j < boxes.size(); ++j) {
        const float dy = std::max({0.f, boxes[i][0] - boxes[j][2], boxes[j][0] - boxes[i][2]});
        const float dx = std::max({0.f, boxes[i][1] - boxes[j][3], boxes[j][1] - boxes[i][3]});
        if (dy <= gap && dx <= gap)
          out.push_back(
              {std::min(boxes[i][0], boxes[j][0]), std::min(boxes[i][1], boxes[j][1]),
               std::max(boxes[i][2], boxes[j][2]), std::max(boxes[i][3], boxes[j][3])});
      }
  }
  std::vector<Box> unique;
  for (const auto &b : out)
    if (std::find(unique.begin(), unique.end(), b) == unique.end())
      unique.push_back(b);
  return unique;
}

/// Box deltas with explicit variances (tflow's EncodeBox/DecodeBox variance argument).
inline std::array<float, 4> EncodeBoxWith(const Box &anchor, const Box &box,
                                          const nn::BoxVariances &variance) {
  return nn::encode_box(anchor, box, variance);
}
inline Box DecodeBoxWith(const Box &anchor, const float *delta,
                         const nn::BoxVariances &variance) {
  return nn::decode_box(anchor, {delta[0], delta[1], delta[2], delta[3]}, variance);
}
inline Box ClipBox(const Box &b, float height, float width) {
  return {std::clamp(b[0], 0.f, height), std::clamp(b[1], 0.f, width),
          std::clamp(b[2], 0.f, height), std::clamp(b[3], 0.f, width)};
}

} // namespace detector
