// Shared pieces of the two-stage (R-CNN family) detection ports (port of
// tflow's common/ml/two_stage.h): a small stride-4 backbone, RoI pooling with
// a two-layer head, the multi-task classification and box-regression loss,
// Fast R-CNN RoI sampling, and region-proposal targets and proposals. The
// RPN head itself (BuildRpn, RpnLoss) lives in common/detection.hpp.
//
// RoI pooling uses nn::crop_and_resize to 14x14 bilinear samples and 2x2 max
// pooling to 7x7 (the max over bins of RoI pooling, with bilinear sampling in
// place of coordinate rounding). Box coordinates map to the feature grid with
// cell i covering pixels [s*i, s*(i+1)) and centred at s*i + s/2. RoIs are
// constants (crop_and_resize differentiates the features only, as in TF).
// RoI sampling takes up to a quarter foreground (IoU >= 0.5 with a target)
// and fills the rest with background of IoU in [0.1, 0.5), then lower
// overlaps; random boxes supplement the background when candidates are few.
// The pyramid (FPN) RoI pooling of tflow is not ported here (fpn/mask_rcnn).
#pragma once

#include <random>

#include "common/detection_scenes.hpp"

namespace detector {

/// CropAndResize's normalized, corner-aligned coordinates on a stride grid.
inline std::array<float, 4> FeatureBox(const Box& b, int stride, int feature_size) {
  auto f = [&](float pixel) { return (pixel - stride / 2.f) / stride / (feature_size - 1); };
  return {f(b[0]), f(b[1]), f(b[2]), f(b[3])};
}

/// Two 3x3 conv/ReLU pairs, each followed by 2x2 max pooling: stride 4, 64 channels.
inline nn::Tensor StrideFourBackbone(nn::ParameterStore& p, const nn::Tensor& images) {
  nn::ConvLayerOptions cfg;
  cfg.init       = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h = nn::conv2d(p, "backbone/conv1", images, 32, 3, cfg);
  h      = nn::max_pool2d(nn::conv2d(p, "backbone/conv2", h, 32, 3, cfg), 2, 2, nn::Padding::Same);
  h      = nn::conv2d(p, "backbone/conv3", h, 64, 3, cfg);
  return nn::max_pool2d(nn::conv2d(p, "backbone/conv4", h, 64, 3, cfg), 2, 2, nn::Padding::Same);
}

/// RoIs of a batch: normalized feature boxes, pixel boxes, image indices and (training) targets.
struct RoiBatch {
  std::vector<std::array<float, 4>> rois, pixels, targets;
  std::vector<int>                  indices, labels;
  std::vector<float>                mask;
};

struct RoiHeadOutputs {
  nn::Tensor logits, probabilities, deltas;  // [K, classes], [K, classes], [K, classes - 1, 4]
};
/// RoI pooling of `features` at normalized `rois` [K, 4] of images `indices`
/// [K], then fc6/fc7 (ReLU), a classifier over `classes` (background last)
/// and class-specific box deltas.
inline RoiHeadOutputs RoiHead(nn::ParameterStore& p, const nn::Tensor& features, const nn::Tensor& rois,
                              const nn::Tensor& indices, int classes, int hidden = 256) {
  const auto pooled = nn::max_pool2d(nn::crop_and_resize(features, rois, indices, {14, 14}), 2, 2, nn::Padding::Valid);
  const std::int64_t channels = features.shape()[3];
  auto f = nn::dense(p, "head/fc6", nn::reshape(pooled, {-1, 7 * 7 * channels}), hidden, nn::Activation::Relu);
  f      = nn::dense(p, "head/fc7", f, hidden, nn::Activation::Relu);
  RoiHeadOutputs h;
  h.logits        = nn::dense(p, "head/classifier", f, classes);
  h.probabilities = nn::softmax(h.logits, -1);
  h.deltas        = nn::reshape(nn::dense(p, "head/regressor", f, 4 * (classes - 1)), {-1, classes - 1, 4});
  return h;
}
inline nn::Tensor RoiTensor(const std::vector<std::array<float, 4>>& boxes) {
  std::vector<float> flat;
  for (const auto& b : boxes) flat.insert(flat.end(), b.begin(), b.end());
  return nn::from_vector(flat, {std::int64_t(boxes.size()), 4});
}
inline nn::Tensor IndexTensor(const std::vector<int>& indices) {
  return nn::from_vector(std::vector<std::int32_t>(indices.begin(), indices.end()));
}
/// Mean cross-entropy over RoIs plus Smooth-L1 regression of the labelled
/// class's deltas on foreground RoIs, normalized by their (at least one) count.
inline nn::Tensor RoiLoss(const RoiHeadOutputs& h, const RoiBatch& b, int classes) {
  const std::int64_t n = std::int64_t(b.labels.size());
  std::vector<float> one_hot(std::size_t(n) * std::size_t(classes), 0.f);
  for (std::int64_t k = 0; k < n; ++k) one_hot[std::size_t(k * classes + b.labels[std::size_t(k)])] = 1;
  const auto targets = nn::from_vector(one_hot, {n, classes});
  const auto cross_entropy = -nn::mean(nn::sum(targets * nn::log_softmax(h.logits, -1), {1}));
  const auto picked = nn::sum(h.deltas * nn::expand_dims(nn::slice(targets, {0, 0}, {n, classes - 1}), 2), {1});
  const auto mask   = nn::from_vector(b.mask);
  const auto regression = nn::sum(nn::sum(SmoothL1(picked - RoiTensor(b.targets)), {1}) * mask) /
                          nn::maximum(nn::sum(mask), nn::scalar(1.0));
  return cross_entropy + regression;
}

/// Sample RoIs of one image (batch position `index`) from candidate boxes
/// (ground truth is added); background is class `background`.
inline void SampleRois(const DetectionScenes& scenes, int row, int index, std::vector<Box> candidates, int total,
                       int background, int stride, int feature_size, std::mt19937_64& random, RoiBatch& batch) {
  const auto& truth = scenes.boxes[std::size_t(row)];
  candidates.insert(candidates.end(), truth.begin(), truth.end());
  std::uniform_real_distribution<float> unit(0, 1);
  for (int extra = 0; extra < total; ++extra) {
    const float h = 10 + 30 * unit(random), w = 10 + 30 * unit(random), y = unit(random) * (scenes.height - h),
                x = unit(random) * (scenes.width - w);
    candidates.push_back({y, x, y + h, x + w});
  }
  std::vector<int>   target(candidates.size(), -1);
  std::vector<float> overlap(candidates.size(), 0);
  for (std::size_t k = 0; k < candidates.size(); ++k)
    for (std::size_t g = 0; g < truth.size(); ++g) {
      const float iou = BoxIoU(candidates[k], truth[g]);
      if (iou > overlap[k]) {
        overlap[k] = iou;
        target[k]  = int(g);
      }
    }
  std::vector<int> foreground, near, far;
  for (std::size_t k = 0; k < candidates.size(); ++k)
    (overlap[k] >= .5f ? foreground : overlap[k] >= .1f ? near : far).push_back(int(k));
  std::shuffle(foreground.begin(), foreground.end(), random);
  std::shuffle(near.begin(), near.end(), random);
  std::shuffle(far.begin(), far.end(), random);
  std::vector<int> chosen(foreground.begin(), foreground.begin() + std::min<std::size_t>(foreground.size(), std::size_t(total / 4)));
  for (int k : near)
    if (int(chosen.size()) < total) chosen.push_back(k);
  for (int k : far)
    if (int(chosen.size()) < total) chosen.push_back(k);
  for (int k : chosen) {
    const auto  uk       = std::size_t(k);
    const bool positive = overlap[uk] >= .5f;
    batch.rois.push_back(FeatureBox(candidates[uk], stride, feature_size));
    batch.pixels.push_back(candidates[uk]);
    batch.indices.push_back(index);
    batch.labels.push_back(positive ? scenes.labels[std::size_t(row)][std::size_t(target[uk])] : background);
    batch.mask.push_back(positive);
    batch.targets.push_back(positive ? EncodeBox(candidates[uk], truth[std::size_t(target[uk])]) : std::array<float, 4>{0, 0, 0, 0});
  }
}

/// Classify and refine boxes [(batch position, pixel box)] of the given rows
/// (features of those rows, batch positions index them); keeps class scores above 0.05.
template <typename HeadFn>
std::vector<Detection> DetectRois(const HeadFn& head, const nn::Tensor& features, const std::vector<int>& rows,
                                  const std::vector<std::pair<int, Box>>& boxes, int classes, int stride, int feature_size) {
  std::vector<Detection> dets;
  if (boxes.empty()) return dets;
  std::vector<std::array<float, 4>> rois;
  std::vector<int>                  indices;
  for (const auto& [index, box] : boxes) {
    rois.push_back(FeatureBox(box, stride, feature_size));
    indices.push_back(index);
  }
  const RoiHeadOutputs out           = head(features, RoiTensor(rois), IndexTensor(indices));
  const auto           probabilities = out.probabilities.template to_vector<float>();
  const auto           deltas        = out.deltas.template to_vector<float>();
  for (std::size_t k = 0; k < boxes.size(); ++k)
    for (int c = 0; c < classes - 1; ++c) {
      const float score = probabilities[k * std::size_t(classes) + std::size_t(c)];
      if (score <= .05f) continue;
      dets.push_back({rows[std::size_t(boxes[k].first)], c, score,
                      DecodeBox(boxes[k].second, deltas.data() + (k * std::size_t(classes - 1) + std::size_t(c)) * 4)});
    }
  return dets;
}

/// RPN training targets: positives at IoU >= 0.7 (and each target's best
/// anchor), negatives below 0.3, `per_image` anchors sampled, at most half
/// positive. Returns labels, weights, positives [N, A] and targets [N, A, 4].
struct RpnTargets {
  nn::Tensor labels, weights, targets, positives;
};
inline RpnTargets MakeRpnTargets(const std::vector<Box>& anchors, const DetectionScenes& scenes,
                                 const std::vector<int>& rows, std::mt19937_64& random, int per_image = 128) {
  const std::size_t  n = rows.size(), count = anchors.size();
  std::vector<float> labels(n * count, 0), weights(n * count, 0), targets(n * count * 4, 0), positives(n * count, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto&      truth = scenes.boxes[std::size_t(rows[i])];
    const auto       match = MatchAnchors(anchors, truth, .7f, .3f);
    std::vector<int> pos, neg;
    for (std::size_t a = 0; a < count; ++a) {
      if (match[a] >= 0) pos.push_back(int(a));
      else if (match[a] == -1) neg.push_back(int(a));
    }
    std::shuffle(pos.begin(), pos.end(), random);
    std::shuffle(neg.begin(), neg.end(), random);
    if (int(pos.size()) > per_image / 2) pos.resize(std::size_t(per_image / 2));
    if (neg.size() > std::size_t(per_image) - pos.size()) neg.resize(std::size_t(per_image) - pos.size());
    for (int a : pos) {
      const std::size_t at = i * count + std::size_t(a);
      labels[at] = weights[at] = positives[at] = 1;
      const auto d = EncodeBox(anchors[std::size_t(a)], truth[std::size_t(match[std::size_t(a)])]);
      for (std::size_t c = 0; c < 4; ++c) targets[at * 4 + c] = d[c];
    }
    for (int a : neg) weights[i * count + std::size_t(a)] = 1;
  }
  const auto N = std::int64_t(n), A = std::int64_t(count);
  return {nn::from_vector(labels, {N, A}), nn::from_vector(weights, {N, A}), nn::from_vector(targets, {N, A, 4}),
          nn::from_vector(positives, {N, A})};
}

/// Decoded, clipped anchors: the best `before` by objectness, NMS 0.7, then `after`.
inline std::vector<Box> RpnProposals(const float* objectness, const float* deltas, const std::vector<Box>& anchors,
                                     int height, int width, int before = 300, int after = 100) {
  const int count = int(anchors.size());
  before          = std::min(before, count);
  std::vector<int> order(static_cast<std::size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + before, order.end(),
                    [&](int a, int b) { return objectness[a] > objectness[b]; });
  std::vector<Box>   boxes;
  std::vector<float> scores;
  for (int k = 0; k < before; ++k) {
    const int a = order[std::size_t(k)];
    auto      b = ClipBox(DecodeBox(anchors[std::size_t(a)], deltas + 4 * a), float(height), float(width));
    if (b[2] - b[0] < 4 || b[3] - b[1] < 4) continue;
    boxes.push_back(b);
    scores.push_back(objectness[a]);
  }
  std::vector<Box> kept;
  for (int k : NonMaxSuppression(boxes, scores, .7f, after)) kept.push_back(boxes[std::size_t(k)]);
  return kept;
}
/// Proposals for every row from one RPN evaluation (objectness [N, A], deltas [N, A, 4]).
inline std::vector<std::vector<Box>> ComputeProposals(const Rpn& rpn, const std::vector<Box>& anchors, int height, int width) {
  const auto objectness = rpn.objectness.to_vector<float>(), deltas = rpn.deltas.to_vector<float>();
  const auto n          = std::size_t(rpn.objectness.shape()[0]);
  std::vector<std::vector<Box>> proposals;
  for (std::size_t i = 0; i < n; ++i)
    proposals.push_back(RpnProposals(objectness.data() + i * anchors.size(), deltas.data() + i * anchors.size() * 4,
                                     anchors, height, width));
  return proposals;
}

}  // namespace detector
