// YOLOv5s v6.0-style CSP backbone, SPPF and bidirectional PAN neck (port of
// tflow's common/nn/yolo_v5.h), CIoU box quality (common/ml/box_losses.h) and
// YOLOv5-style decoding, CIoU regression and ratio/neighbour assignment
// (common/ml/yolo_v5.h).
// https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml
// https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py
// https://arxiv.org/abs/1911.08287
//
// Model: width .5 / depth .33 (backbone widths 32/64/128/256/512, C3 repeats
// 1/2/3/1), a 6x6 stride-two stem with pad 2, serial 5x5 spatial pyramid
// pooling, and C3 blocks after both top-down and bottom-up concatenations. C3
// splits into transformed and bypass branches, then concatenates and fuses
// them. Backbone bottlenecks use identity shortcuts; neck bottlenecks do not.
// BN eps .001, momentum .97 and SiLU. He initialization; no pretrained
// weights. Width division and grayscale inputs give the compact model. Three
// outputs coarse-to-fine (strides 32/16/8) pack [dy, dx, h, w, objectness,
// class logits...] for three priors per cell; prediction biases start at the
// expected objectness log(objects / cells^2) and class prior log(.6 / (C - 1)).
//
// Loss: xy decodes as (2 sigmoid(raw) - .5 + cell) / grid and wh as
// (2 sigmoid(raw))^2 * prior / input size. Priors match when both dimension
// ratios are below four, plus the neighbouring cells that the expanded centre
// range can reach; a fallback best prior covers otherwise unmatched targets.
// Conflicting targets in one slot keep the one with greater shape IoU. Box
// loss is 1 - CIoU, class loss independent BCE, objectness uses detached,
// clipped IoU targets, balanced .4/1/4 from coarse to fine. Box/class/
// objectness weights 5/5/1 (a compact-training adaptation).
//
// CIoU of centre-format boxes (cy, cx, h, w), sizes clamped at epsilon:
// DIoU subtracts squared centre distance over the enclosing diagonal; CIoU
// also subtracts alpha * v with v = 4/pi^2 (atan(w_t/h_t) - atan(w_p/h_p))^2
// and a detached alpha = v / (1 - IoU + v + eps).
#pragma once

#include "common/yolo_v3.hpp"

namespace yolo {

struct BoxQuality {
  nn::Tensor iou, diou, ciou, alpha;
};
inline BoxQuality CompleteBoxQuality(const nn::Tensor& prediction, const nn::Tensor& target, double epsilon = 1e-7) {
  const auto p = nn::unstack(prediction, -1), t = nn::unstack(target, -1);
  const auto e  = nn::scalar(epsilon);
  const auto ph = nn::maximum(p[2], e), pw = nn::maximum(p[3], e), th = nn::maximum(t[2], e), tw = nn::maximum(t[3], e);
  const auto py0 = p[0] - ph * .5, px0 = p[1] - pw * .5, py1 = p[0] + ph * .5, px1 = p[1] + pw * .5;
  const auto ty0 = t[0] - th * .5, tx0 = t[1] - tw * .5, ty1 = t[0] + th * .5, tx1 = t[1] + tw * .5;
  const auto zero         = nn::scalar(0.0);
  const auto ih           = nn::maximum(nn::minimum(py1, ty1) - nn::maximum(py0, ty0), zero);
  const auto iw           = nn::maximum(nn::minimum(px1, tx1) - nn::maximum(px0, tx0), zero);
  const auto intersection = ih * iw;
  BoxQuality q;
  q.iou                    = nn::minimum(nn::scalar(1.0), intersection / nn::maximum(ph * pw + th * tw - intersection, e));
  const auto enclosing_h   = nn::maximum(py1, ty1) - nn::minimum(py0, ty0);
  const auto enclosing_w   = nn::maximum(px1, tx1) - nn::minimum(px0, tx0);
  const auto diagonal      = nn::square(enclosing_h) + nn::square(enclosing_w) + epsilon;
  const auto distance      = nn::square(p[0] - t[0]) + nn::square(p[1] - t[1]);
  q.diou                   = q.iou - distance / diagonal;
  const double pi          = std::acos(-1.);
  const auto   v           = nn::square(nn::atan(tw / th) - nn::atan(pw / ph)) * (4 / (pi * pi));
  q.alpha                  = (v / ((1.0 - q.iou) + (v + epsilon))).detach();
  q.ciou                   = q.diou - q.alpha * v;
  return q;
}

struct YoloV5Config {
  int    classes = 80, width_divisor = 1, reference_size = 640;
  bool   grayscale = false;
  double expected_objects = 8;
  bool   update_stats = true;
};
struct YoloV5Outputs {
  std::array<nn::Tensor, 3> backbone, pyramid, raw;
};
struct CspContext {
  nn::ParameterStore& parameters;
  nn::ParameterStore& state;
  bool                training     = false;
  bool                update_stats = true;
};
inline nn::Tensor CspConv(CspContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out, std::int64_t kernel,
                          std::int64_t stride = 1, std::int64_t padding = -1) {
  resnet::ConvNormConfig norm;
  norm.training     = c.training;
  norm.scale        = true;
  norm.epsilon      = .001;
  norm.momentum     = .97;
  norm.update_stats = c.update_stats;
  const auto y      = resnet::NormalizedConv(c.parameters, c.state, name, x, out, kernel, norm, stride, padding < 0 ? kernel / 2 : padding, false);
  return y * nn::sigmoid(y);
}
struct CspOutputs {
  nn::Tensor transformed, bypass, joined, output;
};
inline CspOutputs CspBlock(CspContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out, int repeats, bool shortcut = true) {
  if (out < 2 || out % 2 || repeats < 1) throw std::invalid_argument("invalid CSP block dimensions");
  const std::int64_t hidden = out / 2;
  CspOutputs         result;
  auto               h = CspConv(c, name + "/left", x, hidden, 1);
  result.bypass        = CspConv(c, name + "/right", x, hidden, 1);
  for (int i = 0; i < repeats; ++i) {
    const auto prefix = name + "/unit" + std::to_string(i + 1);
    auto       branch = CspConv(c, prefix + "/reduce", h, hidden, 1);
    branch            = CspConv(c, prefix + "/expand", branch, hidden, 3);
    h                 = shortcut ? h + branch : branch;
  }
  result.transformed = h;
  result.joined      = nn::concat({h, result.bypass}, 3);
  result.output      = CspConv(c, name + "/fuse", result.joined, out, 1);
  return result;
}
inline nn::Tensor SpatialPyramidFast(CspContext& c, const std::string& name, const nn::Tensor& x, std::int64_t channels) {
  if (channels < 2 || channels % 2) throw std::invalid_argument("SPPF requires an even channel count");
  const auto h      = CspConv(c, name + "/reduce", x, channels / 2, 1);
  const auto first  = nn::max_pool2d(h, 5, 1, nn::Padding::Same);
  const auto second = nn::max_pool2d(first, 5, 1, nn::Padding::Same);
  const auto third  = nn::max_pool2d(second, 5, 1, nn::Padding::Same);
  return CspConv(c, name + "/fuse", nn::concat({h, first, second, third}, 3), channels, 1);
}

inline YoloV5Outputs YoloV5(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image, const YoloV5Config& cfg = {},
                            bool training = false) {
  if (cfg.classes < 1 || cfg.width_divisor < 1 || 16 % cfg.width_divisor || cfg.reference_size < 32 || cfg.reference_size % 32 ||
      !(cfg.expected_objects > 0))
    throw std::invalid_argument("invalid YOLOv5-style configuration");
  YoloV5Outputs result;
  CspContext    c{p, state, training, cfg.update_stats};
  const auto    width = [&](int n) { return std::int64_t(n / cfg.width_divisor); };
  auto          h     = CspConv(c, "stem", image, width(32), 6, 2, 2);
  const int     repeats[] = {1, 2, 3, 1};
  for (int stage = 0; stage < 4; ++stage) {
    const auto name = "backbone/stage" + std::to_string(stage + 1);
    const auto out  = width(64 << stage);
    h               = CspConv(c, name + "/down", h, out, 3, 2);
    h               = CspBlock(c, name + "/c3", h, out, repeats[stage]).output;
    if (stage >= 1) result.backbone[std::size_t(stage - 1)] = h;
  }
  h                    = SpatialPyramidFast(c, "sppf", h, width(512));
  const auto top_route = CspConv(c, "neck/top_route", h, width(256), 1);
  auto upsample = [](const nn::Tensor& x, const nn::Tensor& like) { return nn::resize_nearest(x, {like.shape()[1], like.shape()[2]}); };
  auto middle   = CspBlock(c, "neck/top_down_middle", nn::concat({upsample(top_route, result.backbone[1]), result.backbone[1]}, 3), width(256), 1, false).output;
  const auto middle_route = CspConv(c, "neck/middle_route", middle, width(128), 1);
  const auto fine = CspBlock(c, "neck/top_down_fine", nn::concat({upsample(middle_route, result.backbone[0]), result.backbone[0]}, 3), width(128), 1, false).output;
  auto down     = CspConv(c, "neck/fine_down", fine, width(128), 3, 2);
  middle        = CspBlock(c, "neck/bottom_up_middle", nn::concat({down, middle_route}, 3), width(256), 1, false).output;
  down          = CspConv(c, "neck/middle_down", middle, width(256), 3, 2);
  const auto coarse = CspBlock(c, "neck/bottom_up_coarse", nn::concat({down, top_route}, 3), width(512), 1, false).output;
  result.pyramid    = {coarse, middle, fine};
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  for (int level = 0; level < 3; ++level) {
    const auto name = "prediction" + std::to_string(level + 1);
    if (!p.contains(name + "/b")) {
      std::vector<float> bias(std::size_t(3 * (cfg.classes + 5)), 0);
      const float        cells = float(cfg.reference_size) / float(32 >> level);
      for (int a = 0; a < 3; ++a) {
        bias[std::size_t(a * (cfg.classes + 5) + 4)] = std::log(float(cfg.expected_objects) / (cells * cells));
        for (int label = 0; label < cfg.classes; ++label)
          bias[std::size_t(a * (cfg.classes + 5) + 5 + label)] = cfg.classes == 1 ? 0 : std::log(.6f / (float(cfg.classes) - .999999f));
      }
      p.set(name + "/b", nn::from_vector(bias));
    }
    const auto  logits = nn::conv2d(p, name, result.pyramid[std::size_t(level)], 3 * (cfg.classes + 5), 1, linear);
    const auto& s      = logits.shape();
    result.raw[std::size_t(level)] = nn::reshape(logits, {s[0], s[1], s[2], 3, cfg.classes + 5});
  }
  return result;
}

// ---- loss, targets and decoding ---------------------------------------------------

/// Decoded centre-format boxes [N, c, c, 3, 4] (fractions of the image) of one level.
inline nn::Tensor DecodeYoloV5Boxes(const nn::Tensor& raw, const std::vector<Dimension>& priors, int level) {
  const auto&        s     = raw.shape();
  const std::int64_t cells = s[1];
  std::vector<float> index(static_cast<std::size_t>(cells), 0.f), heights(3), widths(3);
  std::iota(index.begin(), index.end(), 0.f);
  for (std::size_t a = 0; a < 3; ++a) {
    heights[a] = priors[std::size_t(6 - 3 * level) + a][0] / float(32 >> level);
    widths[a]  = priors[std::size_t(6 - 3 * level) + a][1] / float(32 >> level);
  }
  const auto   rows = nn::from_vector(index, {1, cells, 1, 1}), cols = nn::from_vector(index, {1, 1, cells, 1});
  const auto   encoded = nn::unstack(nn::slice(raw, {0, 0, 0, 0, 0}, {s[0], s[1], s[2], 3, 4}), 4);
  const double grid    = double(cells);
  const auto   y = (rows + (nn::sigmoid(encoded[0]) * 2.0 - .5)) / grid, x = (cols + (nn::sigmoid(encoded[1]) * 2.0 - .5)) / grid;
  const auto   h = nn::square(nn::sigmoid(encoded[2]) * 2.0) * nn::from_vector(heights, {1, 1, 1, 3}) / grid;
  const auto   w = nn::square(nn::sigmoid(encoded[3]) * 2.0) * nn::from_vector(widths, {1, 1, 1, 3}) / grid;
  return nn::stack({y, x, h, w}, 4);
}

struct YoloV5LevelTargets {
  std::vector<float> geometry, classes, positive, quality;
  int                class_count = 0;
};
struct YoloV5Targets {
  std::array<YoloV5LevelTargets, 3> levels;
  int                               collisions = 0;
};
inline nn::Tensor YoloV5Loss(const std::array<nn::Tensor, 3>& raw, const YoloV5Targets& t, int classes, const std::vector<Dimension>& priors) {
  if (priors.size() != 9 || classes < 1) throw std::invalid_argument("YOLOv5-style loss requires nine priors and positive classes");
  const double balance[] = {.4, 1, 4};
  nn::Tensor   boxes, labels, counts, objectness;
  for (std::size_t level = 0; level < 3; ++level) {
    const auto&     s    = raw[level].shape();
    const auto&     g    = t.levels[level];
    const nn::Shape grid = {s[0], s[1], s[2], 3};
    const auto      positive = nn::from_vector(g.positive, grid);
    const auto quality = CompleteBoxQuality(DecodeYoloV5Boxes(raw[level], priors, int(level)), nn::from_vector(g.geometry, {s[0], s[1], s[2], 3, 4}));
    const auto box     = nn::sum((1.0 - quality.ciou) * positive);
    const auto predicted_objectness = nn::squeeze(nn::slice(raw[level], {0, 0, 0, 0, 4}, {s[0], s[1], s[2], 3, 1}), {4});
    const auto confidence           = positive * nn::maximum(quality.iou, nn::scalar(0.0)).detach();
    const auto object = nn::mean(LogisticCrossEntropy(predicted_objectness, confidence)) * balance[level];
    const auto logits = nn::slice(raw[level], {0, 0, 0, 0, 5}, {s[0], s[1], s[2], 3, classes});
    const auto label  = nn::sum(nn::sum(LogisticCrossEntropy(logits, nn::from_vector(g.classes, {s[0], s[1], s[2], 3, classes})), {4}) * positive);
    boxes      = level ? boxes + box : box;
    labels     = level ? labels + label : label;
    counts     = level ? counts + nn::sum(positive) : nn::sum(positive);
    objectness = level ? objectness + object : object;
  }
  const auto count = nn::maximum(counts, nn::scalar(1.0));
  return boxes / count * 5.0 + labels / (count * double(classes)) * 5.0 + objectness;
}

inline YoloV5Targets MakeYoloV5Targets(const std::array<nn::Shape, 3>& shapes, const std::vector<Dimension>& priors, const DetectionScenes& scenes,
                                       const std::vector<int>& rows) {
  if (priors.size() != 9 || shapes[0].size() != 5) throw std::invalid_argument("invalid YOLOv5-style target inputs");
  YoloV5Targets      t;
  const std::int64_t batch      = std::int64_t(rows.size());
  const int          resolution = int(shapes[0][1]) * 32;
  for (int level = 0; level < 3; ++level) {
    const auto& s = shapes[std::size_t(level)];
    if (s.size() != 5 || s[0] != batch || s[1] != s[2] || s[3] != 3 || s[4] < 6 || s[1] != shapes[0][1] * (1 << level))
      throw std::invalid_argument("invalid YOLOv5-style prediction dimensions");
    auto&             g     = t.levels[std::size_t(level)];
    const std::size_t slots = std::size_t(batch * s[1] * s[2] * 3);
    g.class_count           = int(s[4]) - 5;
    g.geometry.assign(slots * 4, 0);
    g.classes.assign(slots * std::size_t(g.class_count), 0);
    g.positive.assign(slots, 0);
    g.quality.assign(slots, -1);
  }
  for (std::int64_t i = 0; i < batch; ++i)
    for (std::size_t k = 0; k < scenes.boxes[std::size_t(rows[std::size_t(i)])].size(); ++k) {
      const auto&      b     = scenes.boxes[std::size_t(rows[std::size_t(i)])][k];
      const int        label = scenes.labels[std::size_t(rows[std::size_t(i)])][k];
      const Dimension  size  = {(b[2] - b[0]) / float(scenes.height) * float(resolution), (b[3] - b[1]) / float(scenes.width) * float(resolution)};
      int              best  = 0;
      std::vector<int> matched;
      for (int j = 0; j < 9; ++j) {
        const auto& prior = priors[std::size_t(j)];
        if (DimensionIoU(size, prior) > DimensionIoU(size, priors[std::size_t(best)])) best = j;
        const float ratio = std::max({size[0] / prior[0], prior[0] / size[0], size[1] / prior[1], prior[1] / size[1]});
        if (ratio < 4) matched.push_back(j);
      }
      if (matched.empty()) matched.push_back(best);
      for (int prior : matched) {
        const int level = 2 - prior / 3, a = prior % 3, cells = int(shapes[std::size_t(level)][1]);
        auto&     g     = t.levels[std::size_t(level)];
        if (label < 0 || label >= g.class_count) throw std::invalid_argument("YOLOv5-style class outside range");
        const float y = (b[0] + b[2]) * .5f / float(scenes.height) * float(cells), x = (b[1] + b[3]) * .5f / float(scenes.width) * float(cells);
        const int   r = std::clamp(int(y), 0, cells - 1), c = std::clamp(int(x), 0, cells - 1);
        std::vector<std::pair<int, int>> assign = {{r, c}};
        if (y - float(r) < .5f && r > 0) assign.push_back({r - 1, c});
        if (y - float(r) > .5f && r + 1 < cells) assign.push_back({r + 1, c});
        if (x - float(c) < .5f && c > 0) assign.push_back({r, c - 1});
        if (x - float(c) > .5f && c + 1 < cells) assign.push_back({r, c + 1});
        for (const auto& [row, col] : assign) {
          const std::size_t slot    = std::size_t(((i * cells + row) * cells + col) * 3 + a);
          const float       quality = DimensionIoU(size, priors[std::size_t(prior)]);
          if (g.quality[slot] >= 0) {
            ++t.collisions;
            if (quality <= g.quality[slot]) continue;
          }
          g.quality[slot]       = quality;
          const float encoded[] = {(b[0] + b[2]) * .5f / float(scenes.height), (b[1] + b[3]) * .5f / float(scenes.width),
                                   (b[2] - b[0]) / float(scenes.height), (b[3] - b[1]) / float(scenes.width)};
          for (std::size_t d = 0; d < 4; ++d) g.geometry[slot * 4 + d] = encoded[d];
          for (int j = 0; j < g.class_count; ++j) g.classes[slot * std::size_t(g.class_count) + std::size_t(j)] = j == label;
          g.positive[slot] = 1;
        }
      }
    }
  return t;
}

inline Box DecodeYoloV5(const float* raw, int row, int col, int cells, const Dimension& prior, int stride, int height, int width) {
  auto        sigmoid = [](float x) { return float(1. / (1. + std::exp(-double(x)))); };
  const float y = (float(row) + 2 * sigmoid(raw[0]) - .5f) / float(cells) * float(height);
  const float x = (float(col) + 2 * sigmoid(raw[1]) - .5f) / float(cells) * float(width);
  const float rh = 2 * sigmoid(raw[2]), rw = 2 * sigmoid(raw[3]);
  const float h = rh * rh * prior[0] / float(stride * cells) * float(height), w = rw * rw * prior[1] / float(stride * cells) * float(width);
  return {y - h / 2, x - w / 2, y + h / 2, x + w / 2};
}
/// Inference in chunks of 32 with `forward(images)` returning the three raw heads; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection> DetectYoloV5(const Forward& forward, const std::vector<Dimension>& priors, const DetectionScenes& scenes,
                                              const std::vector<int>& rows) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> result;
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
              const std::size_t slot = ((i * std::size_t(cells) + std::size_t(r)) * std::size_t(cells) + std::size_t(c)) * 3 + std::size_t(a);
              const float*      v    = values.data() + slot * std::size_t(classes + 5);
              const auto b = detector::ClipBox(DecodeYoloV5(v, r, c, cells, priors[std::size_t(6 - 3 * level + a)], stride, scenes.height, scenes.width),
                                               float(scenes.height), float(scenes.width));
              if (!(detector::BoxArea(b) > 0)) continue;
              const float objectness = float(1. / (1. + std::exp(-double(v[4]))));
              for (int label = 0; label < classes; ++label) {
                const float score = objectness * probabilities[slot * std::size_t(classes) + std::size_t(label)];
                if (score > .01f) result.push_back({chunk[i], label, score, b});
              }
            }
    }
  }
  return detector::ClassNms(result, .45f);
}

}  // namespace yolo
