// YOLACT: shared FPN predictions and independent linear-combination prototypes
// (port of tflow's common/nn/yolact.h and common/ml/yolact.h).
// https://arxiv.org/abs/1904.02689
// https://github.com/dbolya/yolact/blob/master/yolact.py
// https://github.com/dbolya/yolact/tree/master/layers
// ResNet101 C3/C4/C5, bilinear top-down P3-P5, strided P6/P7, shared 3x3
// prediction feature and box/class/tanh-coefficient outputs. Protonet: three
// 3x3/ReLU layers, 2x bilinear resize, 3x3/ReLU, 1x1/ReLU prototypes. Semantic
// auxiliary logits come from P3. Background is the last class, unlike the
// reference's background-first convention. No pretrained weights.
// Teaching adaptations (as tflow): true aspect ratios (not the release's
// legacy square anchor bug), pixel-centre binary target resampling, discrete
// crop-area normalization and no one-pixel crop expansion.
//
// TF -> datamunge: the loss is eager. Targets are host data (YolactTargets);
// 3:1 hard-negative mining ranks the detached background cross-entropies on
// the host (TF TopK/ScatterNd ranks, ties lower index first), so the selected
// negatives are the same constants TF's graph computes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/detection_scenes.hpp"
#include "common/instance_segmentation.hpp"
#include "common/resnet.hpp"

namespace yolact {

namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;

// ---- model (tflow nn/yolact.h) ----------------------------------------------------------------

struct YolactConfig {
  int  size = 550, classes = 80, depth = 101, divisor = 1, width = 256, prototypes = 32;
  bool grayscale = false;
};
struct YolactOutputs {
  nn::Tensor       logits, probabilities, deltas, coefficients, prototypes, semantic;
  std::vector<int> cells;
  int              anchors = 0, proto_side = 0;
};

/// Prediction grid sides of the five levels (strides 8..128) for a square input.
inline std::vector<int> YolactCells(int size) {
  std::vector<int> cells;
  for (int i = 0; i < 5; ++i) cells.push_back((size + (8 << i) - 1) / (8 << i));
  return cells;
}

/// `update_stats` false leaves the BN moving averages untouched in training mode (diagnostic passes).
inline YolactOutputs Yolact(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images,
                            const YolactConfig& cfg = {}, bool training = false, bool update_stats = true) {
  if (cfg.size < 32 || cfg.classes < 1 || cfg.width < 1 || cfg.prototypes < 1 || cfg.divisor < 1 || 64 % cfg.divisor)
    throw std::invalid_argument("invalid YOLACT configuration");
  YolactOutputs        out;
  resnet::ResNetConfig rc;
  rc.depth              = cfg.depth;
  rc.width_divisor      = cfg.divisor;
  rc.include_classifier = false;
  rc.stride_on_3x3      = true;
  rc.update_stats       = update_stats;
  const auto backbone = resnet::ResNet(p, state, cfg.grayscale ? nn::tile(images, {1, 1, 1, 3}) : images, rc, training);
  const std::int64_t      w = cfg.width;
  std::vector<nn::Tensor> inner(3), features(3);
  nn::ConvLayerOptions    linear;
  linear.init     = nn::init::he_uniform();
  auto relu       = linear;
  relu.activation = nn::Activation::Relu;
  for (int i = 2; i >= 0; --i) {
    auto x = nn::conv2d(p, "fpn/lateral" + std::to_string(i), backbone.stages[std::size_t(i + 1)], w, 1, linear);
    if (i < 2) x = x + nn::resize_bilinear(inner[std::size_t(i + 1)], {x.shape()[1], x.shape()[2]});
    inner[std::size_t(i)]    = x;
    features[std::size_t(i)] = nn::conv2d(p, "fpn/smooth" + std::to_string(i), x, w, 3, relu);
  }
  auto down    = linear;
  down.stride  = 2;
  down.padding = nn::Padding::Valid;
  for (int i = 0; i < 2; ++i)
    features.push_back(nn::conv2d(p, "fpn/down" + std::to_string(i),
                                  nn::pad(features.back(), {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), w, 3, down));
  out.cells = YolactCells(cfg.size);
  std::vector<nn::Tensor> logits, deltas, coefficients;
  for (int i = 0; i < 5; ++i) {
    const int cells = out.cells[std::size_t(i)], count = 3 * cells * cells;
    if (features[std::size_t(i)].shape()[1] != cells || features[std::size_t(i)].shape()[2] != cells)
      throw std::invalid_argument("YOLACT feature grid does not match the configured input size");
    out.anchors += count;
    const auto x = nn::conv2d(p, "prediction/shared", features[std::size_t(i)], w, 3, relu);
    logits.push_back(nn::reshape(nn::conv2d(p, "prediction/classes", x, 3 * (cfg.classes + 1), 3, linear),
                                 {-1, count, cfg.classes + 1}));
    deltas.push_back(nn::reshape(nn::conv2d(p, "prediction/boxes", x, 12, 3, linear), {-1, count, 4}));
    coefficients.push_back(nn::reshape(
        nn::tanh(nn::conv2d(p, "prediction/coefficients", x, 3 * cfg.prototypes, 3, linear)), {-1, count, cfg.prototypes}));
  }
  out.logits        = nn::concat(logits, 1);
  out.deltas        = nn::concat(deltas, 1);
  out.coefficients  = nn::concat(coefficients, 1);
  out.probabilities = nn::softmax(out.logits, -1);
  nn::Tensor proto  = features[0];
  for (int i = 0; i < 3; ++i) proto = nn::conv2d(p, "proto/conv" + std::to_string(i), proto, w, 3, relu);
  out.proto_side = 2 * out.cells[0];
  proto          = nn::resize_bilinear(proto, {out.proto_side, out.proto_side});
  proto          = nn::conv2d(p, "proto/after_resize", proto, w, 3, relu);
  out.prototypes = nn::conv2d(p, "proto/output", proto, cfg.prototypes, 1, relu);
  out.semantic   = nn::conv2d(p, "semantic", features[0], cfg.classes, 1, linear);
  return out;
}

// ---- training targets and loss (tflow ml/yolact.h) ------------------------------------------

/// prototypes [n, h, w, k] . coefficients [n, k] -> mask logits [n, h, w].
inline nn::Tensor LinearMasks(const nn::Tensor& prototypes, const nn::Tensor& coefficients) {
  return nn::sum(prototypes * nn::expand_dims(nn::expand_dims(coefficients, 1), 1), {3});
}

/// Mean over masks of crop-area-normalized binary cross-entropy (zero for no masks).
inline nn::Tensor CroppedMaskLoss(const nn::Tensor& logits, const nn::Tensor& targets, const nn::Tensor& crop) {
  if (logits.shape()[0] == 0) return nn::scalar(0.0f);
  const auto ce       = nn::softplus(logits) - logits * targets;
  const auto per_mask = nn::sum(ce * crop, {1, 2}) / nn::maximum(nn::sum(crop, {1, 2}), nn::scalar_like(1.0, logits));
  return nn::sum(per_mask) / double(std::max<std::int64_t>(logits.shape()[0], 1));
}

/// SSD matching, then anchors with best IoU >= .4 that are not positive are ignored (-2).
inline std::vector<int> YolactMatching(const std::vector<Box>& anchors, const std::vector<Box>& truth) {
  auto match = detector::MatchSsd(anchors, truth);
  for (std::size_t a = 0; a < match.size(); ++a)
    if (match[a] < 0) {
      float best = 0;
      for (const auto& box : truth) best = std::max(best, detector::BoxIoU(anchors[a], box));
      if (best >= .4f) match[a] = -2;
    }
  return match;
}

inline std::vector<Box> YolactAnchors(const YolactOutputs& net, int size, bool compact) {
  std::vector<Box>         out;
  const std::vector<float> scales =
      compact ? std::vector<float>{8, 16, 24, 40, 56} : std::vector<float>{24, 48, 96, 192, 384};
  for (int i = 0; i < 5; ++i) {
    const int  cells = net.cells[std::size_t(i)];
    const auto level = detector::AnchorGrid(cells, cells, float(size) / float(cells), {scales[std::size_t(i)]}, {1, .5f, 2});
    out.insert(out.end(), level.begin(), level.end());
  }
  return out;
}

/// Host targets of one batch (tflow's placeholders).
struct YolactTargets {
  int                              batch = 0, anchors = 0, side = 0, cells = 0, classes = 0;
  std::vector<std::int32_t>        labels;  // [B, A]: class, `classes` = background, -1 ignored
  std::vector<float>               deltas;  // [B, A, 4]
  std::vector<std::pair<int, int>> pairs;   // (batch image, anchor) of each selected positive mask
  std::vector<float>               mask_targets, crop;  // [n, side, side]
  std::vector<float>               semantic;            // [B, cells, cells, classes]
};

struct YolactLoss {
  nn::Tensor        loss, mask_loss;
  std::vector<char> negatives;  // [B, A] mined hard negatives
};

inline YolactLoss BuildYolactLoss(const YolactOutputs& net, const YolactTargets& t) {
  const int B = t.batch, A = t.anchors, C = t.classes, h = t.side;
  if (net.anchors != A || net.logits.shape()[0] != B) throw std::invalid_argument("YOLACT targets do not match outputs");
  std::vector<float> positive(std::size_t(B) * A, 0.f), onehot(std::size_t(B) * A * (C + 1), 0.f);
  for (std::size_t i = 0; i < positive.size(); ++i) {
    const int label = t.labels[i];
    positive[i]     = label >= 0 && label < C ? 1.f : 0.f;
    if (label >= 0 && label <= C) onehot[i * std::size_t(C + 1) + std::size_t(label)] = 1.f;
  }
  const nn::Tensor mask = nn::from_vector(positive, {B, A});
  const nn::Tensor logp = nn::log_softmax(net.logits, -1);
  const nn::Tensor ce   = -nn::sum(nn::from_vector(onehot, {B, A, C + 1}) * logp, {2});
  // 3:1 hard negatives among background anchors, ranked by detached cross-entropy.
  YolactLoss out;
  out.negatives.assign(std::size_t(B) * A, 0);
  const auto         ce_values = ce.detach().to_vector<float>();
  std::vector<float> chosen(std::size_t(B) * A, 0.f);
  for (int b = 0; b < B; ++b) {
    std::vector<float> score(static_cast<std::size_t>(A));
    int                positives = 0, backgrounds = 0;
    for (int a = 0; a < A; ++a) {
      const std::size_t i = std::size_t(b) * A + std::size_t(a);
      const bool        background = t.labels[i] == C;
      score[std::size_t(a)]        = background ? ce_values[i] : -1e9f;
      positives += positive[i] > 0;
      backgrounds += background;
    }
    std::vector<int> order(static_cast<std::size_t>(A));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return score[std::size_t(x)] > score[std::size_t(y)]; });
    const int wanted = std::min(3 * positives, backgrounds);
    for (int r = 0; r < wanted; ++r) {
      const std::size_t i = std::size_t(b) * A + std::size_t(order[std::size_t(r)]);
      if (t.labels[i] == C) {
        out.negatives[i] = 1;
        chosen[i]        = 1.f;
      }
    }
  }
  const nn::Tensor cls = nn::sum(ce * (mask + nn::from_vector(chosen, {B, A})));
  const nn::Tensor box =
      nn::sum(nn::sum(detector::SmoothL1(net.deltas - nn::from_vector(t.deltas, {B, A, 4})), {2}) * mask);
  const nn::Tensor detection = (cls + 1.5 * box) / std::max(1.0, double(std::accumulate(positive.begin(), positive.end(), 0.f)));
  const int        n = int(t.pairs.size());
  if (n > 0) {
    std::vector<std::int64_t> index, rows;
    for (const auto& [image, anchor] : t.pairs) {
      index.push_back(image);
      index.push_back(anchor);
      rows.push_back(image);
    }
    const nn::Tensor selected = nn::gather_nd(net.coefficients, nn::from_vector(index, {n, 2}));
    const nn::Tensor proto    = nn::gather(net.prototypes, nn::from_vector(rows, {n}), 0);
    out.mask_loss = CroppedMaskLoss(LinearMasks(proto, selected), nn::from_vector(t.mask_targets, {n, h, h}),
                                    nn::from_vector(t.crop, {n, h, h}));
  } else
    out.mask_loss = nn::scalar(0.0f);
  const nn::Tensor semantic =
      nn::mean(nn::softplus(net.semantic) - net.semantic * nn::from_vector(t.semantic, {B, t.cells, t.cells, C}));
  out.loss = detection + 6.125 * out.mask_loss + semantic;
  return out;
}

/// Unlike greedy NMS, a suppressed higher-scoring box can suppress another
/// candidate: the upper-triangular all-pairs rule, per image/class.
inline std::vector<int> FastNms(const std::vector<Detection>& detections, float threshold = .5f, int top_k = 200,
                                int limit = 100) {
  if (!(threshold >= 0 && threshold <= 1) || top_k < 0 || limit < 0) throw std::invalid_argument("invalid Fast NMS configuration");
  std::map<std::pair<int, int>, std::vector<int>> groups;
  for (std::size_t i = 0; i < detections.size(); ++i) groups[{detections[i].image, detections[i].label}].push_back(int(i));
  std::map<int, std::vector<int>> images;
  for (auto& [key, ids] : groups) {
    std::stable_sort(ids.begin(), ids.end(),
                     [&](int a, int b) { return detections[std::size_t(a)].score > detections[std::size_t(b)].score; });
    if (int(ids.size()) > top_k) ids.resize(std::size_t(top_k));
    for (std::size_t j = 0; j < ids.size(); ++j) {
      float best = 0;
      for (std::size_t i = 0; i < j; ++i)
        best = std::max(best, detector::BoxIoU(detections[std::size_t(ids[i])].box, detections[std::size_t(ids[j])].box));
      if (best <= threshold) images[key.first].push_back(ids[j]);
    }
  }
  std::vector<int> out;
  for (auto& [image, ids] : images) {
    std::stable_sort(ids.begin(), ids.end(),
                     [&](int a, int b) { return detections[std::size_t(a)].score > detections[std::size_t(b)].score; });
    if (int(ids.size()) > limit) ids.resize(std::size_t(limit));
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

/// Targets of `rows` (tflow FeedYolactTargets): matched labels/deltas, at most
/// 32 randomly selected positive masks per image, pixel-centre mask/crop
/// targets on the prototype grid and the class-specific semantic union.
inline YolactTargets MakeYolactTargets(const YolactOutputs& net, const std::vector<Box>& anchors,
                                       const instance::InstanceScenes& scenes, const std::vector<int>& rows, int classes,
                                       std::mt19937& random) {
  YolactTargets t;
  t.batch = int(rows.size());
  t.anchors = int(anchors.size());
  t.side = net.proto_side;
  t.cells = net.cells[0];
  t.classes = classes;
  const int b = t.batch, a = t.anchors, h = t.side, cells = t.cells;
  t.labels.assign(std::size_t(b) * a, classes);
  t.deltas.assign(std::size_t(b) * a * 4, 0.f);
  t.semantic.assign(std::size_t(b) * cells * cells * classes, 0.f);
  struct Positive {
    int image, anchor, target;
  };
  std::vector<Positive> positives;
  for (int i = 0; i < b; ++i) {
    const int             row   = rows[std::size_t(i)];
    const auto            match = YolactMatching(anchors, scenes.boxes[std::size_t(row)]);
    std::vector<Positive> selected;
    for (int j = 0; j < a; ++j) {
      const std::size_t k = std::size_t(i) * a + std::size_t(j);
      if (match[std::size_t(j)] == -2) t.labels[k] = -1;
      if (match[std::size_t(j)] < 0) continue;
      t.labels[k]      = scenes.labels[std::size_t(row)][std::size_t(match[std::size_t(j)])];
      const auto delta = detector::EncodeBox(anchors[std::size_t(j)], scenes.boxes[std::size_t(row)][std::size_t(match[std::size_t(j)])]);
      for (int d = 0; d < 4; ++d) t.deltas[k * 4 + std::size_t(d)] = delta[std::size_t(d)];
      selected.push_back({i, j, match[std::size_t(j)]});
    }
    std::shuffle(selected.begin(), selected.end(), random);
    if (selected.size() > 32) selected.resize(32);
    positives.insert(positives.end(), selected.begin(), selected.end());
    for (std::size_t k = 0; k < scenes.labels[std::size_t(row)].size(); ++k)
      for (int y = 0; y < cells; ++y)
        for (int x = 0; x < cells; ++x) {
          const int sy = std::min(scenes.height - 1, int((y + .5f) * scenes.height / cells)),
                    sx = std::min(scenes.width - 1, int((x + .5f) * scenes.width / cells));
          if (scenes.instance_masks[std::size_t(row)][k][std::size_t(sy * scenes.width + sx)])
            t.semantic[((std::size_t(i) * cells + std::size_t(y)) * cells + std::size_t(x)) * classes +
                       std::size_t(scenes.labels[std::size_t(row)][k])] = 1;
        }
  }
  const int n = int(positives.size());
  t.mask_targets.assign(std::size_t(n) * h * h, 0.f);
  t.crop.assign(std::size_t(n) * h * h, 0.f);
  for (int i = 0; i < n; ++i) {
    const auto& pos = positives[std::size_t(i)];
    const int   row = rows[std::size_t(pos.image)];
    const auto& box = scenes.boxes[std::size_t(row)][std::size_t(pos.target)];
    t.pairs.push_back({pos.image, pos.anchor});
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < h; ++x) {
        const float       sy = (y + .5f) * scenes.height / h, sx = (x + .5f) * scenes.width / h;
        const std::size_t k  = (std::size_t(i) * h + std::size_t(y)) * h + std::size_t(x);
        t.mask_targets[k] = scenes.instance_masks[std::size_t(row)][std::size_t(pos.target)]
                                                  [std::size_t(std::min(scenes.height - 1, int(sy)) * scenes.width +
                                                               std::min(scenes.width - 1, int(sx)))];
        t.crop[k] = sy >= box[0] && sy < box[2] && sx >= box[1] && sx < box[3];
      }
  }
  return t;
}

/// Inference: class scores > .05 on decoded clipped boxes, Fast NMS, then
/// sigmoid prototype masks inside each box pasted to the scene grid.
/// `infer(images)` returns inference-mode outputs; rows are processed in chunks of 8.
template <typename InferFn>
std::vector<instance::MaskDetection> DetectYolact(const InferFn& infer, const std::vector<Box>& anchors,
                                                  const instance::InstanceScenes& scenes, const std::vector<int>& rows,
                                                  int classes) {
  nn::NoGradGuard                      guard;
  std::vector<instance::MaskDetection> result;
  for (std::size_t start = 0; start < rows.size(); start += 8) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 8)));
    const YolactOutputs    net           = infer(detector::SceneImages(scenes, chunk));
    const auto             probabilities = net.probabilities.to_vector<float>();
    const auto             deltas        = net.deltas.to_vector<float>();
    const auto             coefficients  = net.coefficients.to_vector<float>();
    const auto             prototypes    = net.prototypes.to_vector<float>();
    const std::size_t      A             = anchors.size();
    std::vector<Detection>           candidates;
    std::vector<std::pair<int, int>> sources;
    for (std::size_t i = 0; i < chunk.size(); ++i)
      for (std::size_t a = 0; a < A; ++a) {
        auto box = detector::DecodeBox(anchors[a], deltas.data() + (i * A + a) * 4);
        box      = {std::clamp(box[0], 0.f, float(scenes.height)), std::clamp(box[1], 0.f, float(scenes.width)),
                    std::clamp(box[2], 0.f, float(scenes.height)), std::clamp(box[3], 0.f, float(scenes.width))};
        if (detector::BoxArea(box) <= 0) continue;
        for (int c = 0; c < classes; ++c) {
          const float score = probabilities[(i * A + a) * std::size_t(classes + 1) + std::size_t(c)];
          if (score > .05f) {
            candidates.push_back({chunk[i], c, score, box});
            sources.push_back({int(i), int(a)});
          }
        }
      }
    const int h = net.proto_side, k = int(net.coefficients.shape()[2]);
    for (int index : FastNms(candidates)) {
      const auto [image, anchor] = sources[std::size_t(index)];
      const auto&        detection = candidates[std::size_t(index)];
      std::vector<float> mask(std::size_t(h) * h, 0.f);
      const float*       proto = prototypes.data() + std::size_t(image) * h * h * k;
      const float*       coeff = coefficients.data() + (std::size_t(image) * A + std::size_t(anchor)) * std::size_t(k);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < h; ++x) {
          const float py = (y + .5f) * scenes.height / h, px = (x + .5f) * scenes.width / h;
          if (py < detection.box[0] || py >= detection.box[2] || px < detection.box[1] || px >= detection.box[3]) continue;
          double value = 0;
          for (int j = 0; j < k; ++j) value += proto[(std::size_t(y) * h + std::size_t(x)) * std::size_t(k) + std::size_t(j)] * coeff[j];
          check::Require(std::isfinite(value), "finite assembled prototype mask");
          mask[std::size_t(y) * h + std::size_t(x)] = float(1 / (1 + std::exp(-std::clamp(value, -80., 80.))));
        }
      result.push_back({detection, instance::PasteMask(mask.data(), h, 1, 0, {0, 0, float(scenes.height), float(scenes.width)},
                                                       scenes.height, scenes.width)});
    }
  }
  return result;
}

}  // namespace yolact
