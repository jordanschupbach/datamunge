// RetinaNet: ResNet feature pyramid, shared four-convolution subnets, sigmoid
// focal loss, anchor targets and decoding (port of tflow's
// common/nn/retinanet.h and common/ml/retinanet.h).
// https://arxiv.org/abs/1708.02002
//
// Model: ResNet C3/C4/C5 at strides 8/16/32, lateral/top-down P3/P4/P5, P6
// from a stride-two 3x3 on C5, and P7 from ReLU(P6) plus stride-two 3x3.
// Classification and regression each use four shared 3x3/ReLU layers; all
// five levels share their respective predictor. There are nine priors per
// cell. Classification bias starts at logit(.01); subnet kernels have normal
// stddev .01, as in the focal-loss paper. No pretrained weights are loaded.
// Full defaults use ResNet-50, 256 FPN/head channels and 80 classes.
//
// Loss: FL = alpha_t * (1-p_t)^gamma * stable BCE, defaults alpha=.25,
// gamma=2, the modulator computed in log space. Positives are SSD-matched
// (distinct forced best matches, then IoU >= .5), anchors with best IoU in
// [.4, .5) are ignored, the rest are negatives. Focal and Smooth-L1
// (beta=1/9) box losses are normalized by the number of positive anchors.
// Box deltas use unit variances.
#pragma once

#include "common/detection_scenes.hpp"
#include "common/fpn.hpp"
#include "common/resnet.hpp"

namespace retinanet {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

inline nn::Tensor RetinaConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out,
                             bool relu = true, double initial_bias = 0) {
  const std::int64_t in = x.shape()[3];
  const auto         w  = p.get(name + "/W", {3, 3, in, out}, nn::init::normal(.01));
  const auto         y  = nn::bias_add(nn::conv2d(x, w), p.get(name + "/b", {out}, nn::init::constant(initial_bias)));
  return relu ? nn::relu(y) : y;
}

/// Symmetric zero padding of the spatial axes.
inline nn::Tensor PadSpatial(const nn::Tensor& x, std::int64_t pad) { return nn::pad(x, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}); }

/// Detector outputs shared by RetinaNet and EfficientDet.
struct RetinaNetOutputs {
  std::vector<nn::Tensor> features;
  std::vector<int>        cells;
  nn::Tensor              logits, probabilities, deltas;  // [N, A, classes], [N, A, classes], [N, A, 4]
  int                     anchors = 0;
};

struct RetinaNetConfig {
  int  classes = 80, image_size = 640, backbone_depth = 50, width_divisor = 1, pyramid_width = 256;
  bool grayscale = false;
  bool update_stats = true;
};

/// ResNet backbone and P3..P7 (shared with FCOS).
inline std::vector<nn::Tensor> RetinaPyramid(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                             int depth, int width_divisor, std::int64_t width, bool grayscale,
                                             bool training, bool update_stats) {
  resnet::ResNetConfig backbone;
  backbone.depth              = depth;
  backbone.width_divisor      = width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3      = true;
  backbone.update_stats       = update_stats;
  const auto net = resnet::ResNet(p, state, grayscale ? nn::tile(image, {1, 1, 1, 3}) : image, backbone, training);
  auto features  = fpn::FeaturePyramid(p, "fpn", {net.stages[1], net.stages[2], net.stages[3]}, width, false).levels;
  nn::ConvLayerOptions down;
  down.stride  = 2;
  down.padding = nn::Padding::Valid;
  down.init    = nn::init::he_uniform();
  features.push_back(nn::conv2d(p, "fpn/p6", PadSpatial(net.stages[3], 1), width, 3, down));
  features.push_back(nn::conv2d(p, "fpn/p7", PadSpatial(nn::relu(features.back()), 1), width, 3, down));
  return features;
}

inline RetinaNetOutputs RetinaNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                  const RetinaNetConfig& cfg = {}, bool training = false) {
  if (cfg.classes < 1 || cfg.image_size < 32 || cfg.image_size % 32 || cfg.pyramid_width < 1 || cfg.width_divisor < 1 ||
      64 % cfg.width_divisor)
    throw std::invalid_argument("invalid RetinaNet configuration");
  RetinaNetOutputs result;
  result.features = RetinaPyramid(p, state, image, cfg.backbone_depth, cfg.width_divisor, cfg.pyramid_width,
                                  cfg.grayscale, training, cfg.update_stats);
  std::vector<nn::Tensor> logits, deltas;
  for (int level = 0; level < 5; ++level) {
    const int stride = 8 << level, cells = (cfg.image_size + stride - 1) / stride, count = cells * cells * 9;
    if (result.features[std::size_t(level)].shape()[1] != cells)
      throw std::invalid_argument("RetinaNet image_size does not match the input images");
    result.cells.push_back(cells);
    result.anchors += count;
    nn::Tensor classification = result.features[std::size_t(level)], regression = classification;
    for (int layer = 0; layer < 4; ++layer) {
      classification = RetinaConv(p, "classification/conv" + std::to_string(layer + 1), classification, cfg.pyramid_width);
      regression     = RetinaConv(p, "regression/conv" + std::to_string(layer + 1), regression, cfg.pyramid_width);
    }
    logits.push_back(nn::reshape(RetinaConv(p, "classification/output", classification, 9 * cfg.classes, false, std::log(.01 / .99)),
                                 {-1, count, cfg.classes}));
    deltas.push_back(nn::reshape(RetinaConv(p, "regression/output", regression, 36, false), {-1, count, 4}));
  }
  result.logits        = nn::concat(logits, 1);
  result.deltas        = nn::concat(deltas, 1);
  result.probabilities = nn::sigmoid(result.logits);
  return result;
}

// ---- loss, anchors, targets and decoding -------------------------------------------

inline nn::Tensor SigmoidFocalLoss(const nn::Tensor& logits, const nn::Tensor& target, double alpha = .25, double gamma = 2) {
  if (!(alpha >= 0 && alpha <= 1) || !(gamma >= 0) || !std::isfinite(gamma)) throw std::invalid_argument("invalid focal-loss parameters");
  const auto at        = alpha * target + (1 - alpha) * (1.0 - target);
  const auto bce       = nn::softplus(logits) - logits * target;
  const auto sign      = target * 2.0 - 1.0;
  const auto modulator = nn::exp(-gamma * nn::softplus(sign * logits));
  return at * modulator * bce;
}

struct RetinaTargets {
  nn::Tensor classes, positive, weights, boxes;  // [N, A, C], [N, A], [N, A], [N, A, 4]
};
inline nn::Tensor RetinaLoss(const nn::Tensor& logits, const nn::Tensor& deltas, const RetinaTargets& t) {
  const auto   focal      = nn::sum(nn::sum(SigmoidFocalLoss(logits, t.classes), {2}) * t.weights);
  const auto   difference = deltas - t.boxes;
  const auto   distance   = nn::abs(difference);
  const double beta       = 1. / 9;
  const auto   smooth     = nn::where(nn::less(distance, beta), nn::square(difference) * (.5 / beta), distance - .5 * beta);
  const auto   regression = nn::sum(nn::sum(smooth, {2}) * t.positive);
  return (focal + regression) / nn::maximum(nn::sum(t.positive), nn::scalar(1.0));
}

inline std::vector<Box> RetinaAnchors(const std::vector<int>& cells, float first_size = 32) {
  std::vector<Box> result;
  for (std::size_t level = 0; level < cells.size(); ++level) {
    const float size  = std::ldexp(first_size, int(level));
    const auto  boxes = detector::AnchorGrid(cells[level], cells[level], float(8 << level),
                                             {size, size * std::cbrt(2.f), size * std::cbrt(4.f)}, {.5f, 1, 2});
    result.insert(result.end(), boxes.begin(), boxes.end());
  }
  return result;
}

inline RetinaTargets MakeRetinaTargets(const std::vector<Box>& anchors, const DetectionScenes& scenes,
                                       const std::vector<int>& rows, int classes) {
  const std::size_t  batch = rows.size(), count = anchors.size();
  std::vector<float> labels(batch * count * std::size_t(classes), 0), positive(batch * count, 0), weights(batch * count, 1),
      boxes(batch * count * 4, 0);
  for (std::size_t i = 0; i < batch; ++i) {
    const auto& truth    = scenes.boxes[std::size_t(rows[i])];
    const auto  matching = detector::MatchSsd(anchors, truth);
    for (std::size_t a = 0; a < count; ++a) {
      const std::size_t at = i * count + a;
      if (matching[a] >= 0) {
        const int label = scenes.labels[std::size_t(rows[i])][std::size_t(matching[a])];
        if (label < 0 || label >= classes) throw std::invalid_argument("RetinaNet target outside class range");
        labels[at * std::size_t(classes) + std::size_t(label)] = positive[at] = 1;
        const auto delta = detector::EncodeBoxWith(anchors[a], truth[std::size_t(matching[a])], {1, 1, 1, 1});
        for (std::size_t d = 0; d < 4; ++d) boxes[at * 4 + d] = delta[d];
      } else {
        float best = 0;
        for (const auto& b : truth) best = std::max(best, detector::BoxIoU(anchors[a], b));
        if (best >= .4f) weights[at] = 0;
      }
    }
  }
  const auto N = std::int64_t(batch), A = std::int64_t(count);
  return {nn::from_vector(labels, {N, A, classes}), nn::from_vector(positive, {N, A}), nn::from_vector(weights, {N, A}),
          nn::from_vector(boxes, {N, A, 4})};
}

/// Top 200 anchors above .01 per image and class, decoded and clipped, then class-wise NMS .5.
inline std::vector<detector::Detection> DecodeRetina(const nn::Tensor& probabilities, const nn::Tensor& deltas,
                                                     const std::vector<Box>& anchors, const DetectionScenes& scenes,
                                                     const std::vector<int>& rows, int classes) {
  const auto                       P = probabilities.to_vector<float>(), D = deltas.to_vector<float>();
  const std::size_t                A = anchors.size();
  std::vector<detector::Detection> result;
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (int label = 0; label < classes; ++label) {
      std::vector<std::pair<float, int>> scores;
      for (std::size_t a = 0; a < A; ++a) {
        const float score = P[(i * A + a) * std::size_t(classes) + std::size_t(label)];
        if (score > .01f) scores.push_back({score, int(a)});
      }
      const std::size_t count = std::min<std::size_t>(200, scores.size());
      std::partial_sort(scores.begin(), scores.begin() + std::ptrdiff_t(count), scores.end(), std::greater<std::pair<float, int>>());
      for (std::size_t k = 0; k < count; ++k) {
        const auto a = std::size_t(scores[k].second);
        const auto b = detector::ClipBox(detector::DecodeBoxWith(anchors[a], D.data() + (i * A + a) * 4, {1, 1, 1, 1}),
                                         float(scenes.height), float(scenes.width));
        if (detector::BoxArea(b) > 0) result.push_back({rows[i], label, scores[k].first, b});
      }
    }
  return result;
}

/// Inference in chunks of 32 scenes with `forward(images)` returning RetinaNetOutputs; class-wise NMS .5.
template <typename Forward>
std::vector<detector::Detection> DetectRetina(const Forward& forward, const std::vector<Box>& anchors,
                                              const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const RetinaNetOutputs net   = forward(detector::SceneImages(scenes, chunk));
    const auto             found = DecodeRetina(net.probabilities, net.deltas, anchors, scenes, chunk, classes);
    result.insert(result.end(), found.begin(), found.end());
  }
  return detector::ClassNms(result, .5f);
}

}  // namespace retinanet
