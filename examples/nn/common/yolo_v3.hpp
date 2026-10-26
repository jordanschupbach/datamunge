// YOLO v3: Darknet-53 residual features and three top-down detection heads
// (port of tflow's common/nn/yolo_v3.h), with its three-scale assignment,
// logistic losses and decoding (common/ml/yolo_v3.h).
// https://arxiv.org/abs/1804.02767
// https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
//
// Model: Darknet's five stages contain 1/2/8/8/4 residual units. Downsampling
// is a padded stride-two 3x3 convolution; each unit is 1x1/3x3 plus identity.
// All feature convolutions use BN (eps 1e-5) and leaky ReLU .1. Five
// alternating 1x1/3x3 layers form each route; a 3x3 and a linear 1x1 predict
// three anchors. Reduced routes are nearest-upsampled and concatenated with
// earlier features. Outputs are coarse-to-fine (strides 32/16/8), laid out as
// [dy, dx, log(h/prior_h), log(w/prior_w), objectness, class logits...].
// Width division and grayscale inputs adapt the topology for digit scenes.
//
// Loss: the nine priors are sorted by area, three per scale. Each target
// selects exactly one prior globally, then its centre cell at that scale.
// Unassigned predictions with current IoU > .5 are ignored for objectness.
// Independent class BCE; coordinates use half squared error weighted by
// 2 - relative target area. All three heads are summed and normalized by the
// total number of positives. Priors are in input pixels at the base size;
// multi-scale training changes their relative image size.
#pragma once

#include "common/yolo_v2.hpp"

namespace yolo {

/// softplus(x) - x * t: stable BCE with the correct derivative at zero.
inline nn::Tensor LogisticCrossEntropy(const nn::Tensor& logits, const nn::Tensor& target) {
  return nn::softplus(logits) - logits * target;
}

struct YoloV3Config {
  int    classes = 80, width_divisor = 1;
  bool   grayscale = false;
  double momentum = .9;
  bool   update_stats = true;
};
struct YoloV3Outputs {
  std::array<nn::Tensor, 3> backbone, routes, raw;  // raw coarse-to-fine [N, c, c, 3, 5 + classes]
};

inline YoloV3Outputs YoloV3(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image, const YoloV3Config& cfg = {},
                            bool training = false) {
  if (cfg.classes < 1 || cfg.width_divisor < 1 || 32 % cfg.width_divisor || !(cfg.momentum >= 0 && cfg.momentum < 1))
    throw std::invalid_argument("invalid YOLO v3 configuration");
  YoloV3Outputs result;
  auto conv = [&](const std::string& name, const nn::Tensor& x, std::int64_t out, std::int64_t kernel, std::int64_t stride = 1) {
    return DarknetConv(p, state, name, x, out, kernel, training, cfg.momentum, cfg.update_stats, stride);
  };
  std::int64_t in = 32 / cfg.width_divisor;
  auto         h  = conv("backbone/stem", image, in, 3);
  const int    units[] = {1, 2, 8, 8, 4};
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t out    = (64 << stage) / cfg.width_divisor;
    const std::string  prefix = "backbone/stage" + std::to_string(stage + 1);
    h                         = conv(prefix + "/down", h, out, 3, 2);
    in                        = out;
    for (int unit = 0; unit < units[stage]; ++unit) {
      const auto name   = prefix + "/unit" + std::to_string(unit + 1);
      const auto branch = conv(name + "/reduce", h, in / 2, 1);
      h                 = h + conv(name + "/expand", branch, in, 3);
    }
    if (stage >= 2) result.backbone[std::size_t(stage - 2)] = h;
  }
  for (int level = 0; level < 3; ++level) {
    const std::int64_t width  = (512 >> level) / cfg.width_divisor;
    const auto         prefix = "head" + std::to_string(level + 1);
    if (level > 0) {
      const auto  reduced = conv(prefix + "/route_reduce", result.routes[std::size_t(level - 1)], width, 1);
      const auto& skip    = result.backbone[std::size_t(2 - level)];
      h = nn::concat({nn::resize_nearest(reduced, {skip.shape()[1], skip.shape()[2]}), skip}, 3);
    }
    for (int layer = 0; layer < 5; ++layer) h = conv(prefix + "/conv" + std::to_string(layer + 1), h, layer % 2 ? 2 * width : width, layer % 2 ? 3 : 1);
    result.routes[std::size_t(level)] = h;
    const auto           features     = conv(prefix + "/detect", h, 2 * width, 3);
    nn::ConvLayerOptions linear;
    linear.init                    = nn::init::he_uniform();
    const auto  raw                = nn::conv2d(p, prefix + "/prediction", features, 3 * (cfg.classes + 5), 1, linear);
    const auto& s                  = raw.shape();
    result.raw[std::size_t(level)] = nn::reshape(raw, {s[0], s[1], s[2], 3, cfg.classes + 5});
  }
  return result;
}

struct YoloV3LevelTargets {
  std::vector<float> geometry, classes, positive, objectness_weight, geometry_weight, areas;
  nn::Shape          grid;  // [N, c, c, 3]
  int                class_count = 0;
};
struct YoloV3Targets {
  std::array<YoloV3LevelTargets, 3> levels;
  int                               collisions = 0;
};

inline nn::Tensor YoloV3Loss(const std::array<nn::Tensor, 3>& raw, const YoloV3Targets& t, int classes) {
  nn::Tensor total, positives;
  for (std::size_t level = 0; level < 3; ++level) {
    const auto& s = raw[level].shape();
    const auto& g = t.levels[level];
    const nn::Shape grid = {s[0], s[1], s[2], 3};
    auto part = [&](std::int64_t begin, std::int64_t count) { return nn::slice(raw[level], {0, 0, 0, 0, begin}, {s[0], s[1], s[2], 3, count}); };
    const auto geometry    = nn::concat({nn::sigmoid(part(0, 2)), part(2, 2)}, 4);
    const auto positive    = nn::from_vector(g.positive, grid);
    const auto coordinates = nn::sum(nn::sum(nn::square(geometry - nn::from_vector(g.geometry, {s[0], s[1], s[2], 3, 4})), {4}) *
                                     nn::from_vector(g.geometry_weight, grid)) * .5;
    const auto confidence  = nn::sum(LogisticCrossEntropy(nn::squeeze(part(4, 1), {4}), positive) * nn::from_vector(g.objectness_weight, grid));
    const auto classification =
        nn::sum(nn::sum(LogisticCrossEntropy(part(5, classes), nn::from_vector(g.classes, {s[0], s[1], s[2], 3, classes})), {4}) * positive);
    const auto level_total = coordinates + confidence + classification;
    total                  = level ? total + level_total : level_total;
    positives              = level ? positives + nn::sum(positive) : nn::sum(positive);
  }
  return total / nn::maximum(positives, nn::scalar(1.0));
}

/// Targets from detached three-scale predictions (host values and shapes, coarse-to-fine).
inline YoloV3Targets MakeYoloV3Targets(const std::array<std::vector<float>, 3>& raw, const std::array<nn::Shape, 3>& shapes,
                                       const std::vector<Dimension>& priors, const DetectionScenes& scenes, const std::vector<int>& rows) {
  if (priors.size() != 9) throw std::invalid_argument("YOLO v3 requires three heads and nine priors");
  YoloV3Targets      t;
  const std::int64_t batch = std::int64_t(rows.size());
  for (int level = 0; level < 3; ++level) {
    const auto& s = shapes[std::size_t(level)];
    if (s.size() != 5 || s[0] != batch || s[1] != s[2] || s[3] != 3) throw std::invalid_argument("invalid YOLO v3 prediction shape");
    const int cells = int(s[1]), classes = int(s[4]) - 5;
    if (classes < 1 || cells != shapes[0][1] * (1 << level)) throw std::invalid_argument("YOLO v3 prediction levels must double their grids");
    auto&             g     = t.levels[std::size_t(level)];
    const std::size_t slots = std::size_t(batch * cells * cells * 3);
    g.grid                  = {batch, cells, cells, 3};
    g.class_count           = classes;
    g.geometry.assign(slots * 4, 0);
    g.classes.assign(slots * std::size_t(classes), 0);
    g.positive.assign(slots, 0);
    g.geometry_weight.assign(slots, 0);
    g.objectness_weight.assign(slots, 1);
    g.areas.assign(slots, 0);
    const int stride = 32 >> level;
    for (std::int64_t i = 0; i < batch; ++i)
      for (int r = 0; r < cells; ++r)
        for (int c = 0; c < cells; ++c)
          for (int a = 0; a < 3; ++a) {
            const auto        prior = priors[std::size_t(6 - 3 * level + a)];
            const std::size_t slot  = std::size_t(((i * cells + r) * cells + c) * 3 + a);
            const auto box = DecodeYoloV2(raw[std::size_t(level)].data() + slot * std::size_t(classes + 5), r, c, cells,
                                          {prior[0] / float(stride), prior[1] / float(stride)}, scenes.height, scenes.width);
            float overlap = 0;
            for (const auto& truth : scenes.boxes[std::size_t(rows[std::size_t(i)])]) overlap = std::max(overlap, detector::BoxIoU(box, truth));
            if (overlap > .5f) g.objectness_weight[slot] = 0;
          }
  }
  const int resolution = int(shapes[0][1]) * 32;
  for (std::int64_t i = 0; i < batch; ++i)
    for (std::size_t k = 0; k < scenes.boxes[std::size_t(rows[std::size_t(i)])].size(); ++k) {
      const auto&     box  = scenes.boxes[std::size_t(rows[std::size_t(i)])][k];
      const Dimension size = {(box[2] - box[0]) / float(scenes.height) * float(resolution), (box[3] - box[1]) / float(scenes.width) * float(resolution)};
      int             best = 0;
      for (int j = 1; j < 9; ++j)
        if (DimensionIoU(size, priors[std::size_t(j)]) > DimensionIoU(size, priors[std::size_t(best)])) best = j;
      const int   level = 2 - best / 3, a = best % 3, cells = int(shapes[std::size_t(level)][1]);
      auto&       g     = t.levels[std::size_t(level)];
      const float y = (box[0] + box[2]) * .5f / float(scenes.height) * float(cells), x = (box[1] + box[3]) * .5f / float(scenes.width) * float(cells);
      const int   r = std::clamp(int(y), 0, cells - 1), c = std::clamp(int(x), 0, cells - 1);
      const int   label = scenes.labels[std::size_t(rows[std::size_t(i)])][k];
      if (label < 0 || label >= g.class_count) throw std::invalid_argument("YOLO v3 target label outside class range");
      const std::size_t slot = std::size_t(((i * cells + r) * cells + c) * 3 + a);
      const float       area = size[0] * size[1];
      if (g.areas[slot] > 0) {
        ++t.collisions;
        if (area <= g.areas[slot]) continue;
      }
      g.areas[slot]           = area;
      const float encoded[] = {y - float(r), x - float(c), std::log(size[0] / priors[std::size_t(best)][0]), std::log(size[1] / priors[std::size_t(best)][1])};
      for (std::size_t d = 0; d < 4; ++d) g.geometry[slot * 4 + d] = encoded[d];
      for (int j = 0; j < g.class_count; ++j) g.classes[slot * std::size_t(g.class_count) + std::size_t(j)] = j == label;
      g.positive[slot] = g.objectness_weight[slot] = 1;
      g.geometry_weight[slot]                      = 2 - area / float(resolution * resolution);
    }
  return t;
}

/// Inference in chunks of 32 with `forward(images)` returning the three raw heads; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection> DetectYoloV3(const Forward& forward, const std::vector<Dimension>& priors, const DetectionScenes& scenes,
                                              const std::vector<int>& rows) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int>          chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const std::array<nn::Tensor, 3> raw = forward(detector::SceneImages(scenes, chunk));
    for (int level = 0; level < 3; ++level) {
      const auto& s = raw[std::size_t(level)].shape();
      const int   cells = int(s[1]), classes = int(s[4]) - 5, stride = 32 >> level;
      const auto  values        = raw[std::size_t(level)].to_vector<float>();
      const auto  probabilities = nn::sigmoid(nn::slice(raw[std::size_t(level)], {0, 0, 0, 0, 5}, {s[0], s[1], s[2], 3, classes})).to_vector<float>();
      for (std::size_t i = 0; i < chunk.size(); ++i)
        for (int r = 0; r < cells; ++r)
          for (int c = 0; c < cells; ++c)
            for (int a = 0; a < 3; ++a) {
              const std::size_t slot  = ((i * std::size_t(cells) + std::size_t(r)) * std::size_t(cells) + std::size_t(c)) * 3 + std::size_t(a);
              const float*      v     = values.data() + slot * std::size_t(classes + 5);
              const auto        prior = priors[std::size_t(6 - 3 * level + a)];
              const auto b = detector::ClipBox(DecodeYoloV2(v, r, c, cells, {prior[0] / float(stride), prior[1] / float(stride)}, scenes.height, scenes.width),
                                               float(scenes.height), float(scenes.width));
              if (!(detector::BoxArea(b) > 0)) continue;
              const float objectness = float(1. / (1. + std::exp(-double(v[4]))));
              for (int label = 0; label < classes; ++label) {
                const float score = objectness * probabilities[slot * std::size_t(classes) + std::size_t(label)];
                if (score > .01f) detections.push_back({chunk[i], label, score, b});
              }
            }
    }
  }
  return detector::ClassNms(detections, .45f);
}

}  // namespace yolo
