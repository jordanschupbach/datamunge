// YOLO v2: Darknet-19, batch normalization, anchor prediction and reorg (port
// of tflow's common/nn/yolo_v2.h), plus anchor-based losses, IoU dimension
// clustering, targets and decoding (common/ml/yolo_v2.h).
// https://arxiv.org/abs/1612.08242
//
// Model: Darknet-19 without its classifier keeps all 18 feature
// convolutions, then two 3x3 layers, a reduced fine-feature passthrough
// (space-to-depth), a fusion 3x3 and a biased linear prediction layer. Other
// convolutions use BN (gamma/beta, eps 1e-5, momentum .9) and leaky ReLU .1.
// He initialization; no ImageNet weights. Output per anchor:
// [dy, dx, log(h/prior_h), log(w/prior_w), objectness, classes...]. Inputs are
// square with sides divisible by 32. compact=true (grayscale, width_divisor
// 8) skips the first two pools (stride 8). anchors=3, classes=9418 builds
// YOLO9000's head; the supplied hierarchy defines its labels.
//
// Loss: centres are sigmoid offsets within cells; sizes exponentiate offsets
// from dimension priors (feature-grid units). Each target takes its best
// shape prior and centre cell; unmatched predictions overlapping a target by
// more than .6 IoU are ignored. Positive confidence targets are current IoUs.
// Coordinate SSE (weight 5), sigmoid confidence SSE (background weight .5)
// and class cross-entropy, normalized by positive anchors; collisions keep
// the larger target. With a WordTree, classification-only examples select
// the prediction with the highest objectness-weighted marginal probability of
// their label, supervise that ancestor path and penalize objectness below .3,
// without box targets. All matching/IoU targets come from detached values.
#pragma once

#include "common/detection_scenes.hpp"
#include "common/resnet.hpp"
#include "common/word_tree.hpp"

namespace yolo {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

struct YoloV2Config {
  int    classes = 20, anchors = 5, width_divisor = 1;
  bool   compact = false;
  double momentum = .9;
  bool   update_stats = true;
};
struct YoloV2Outputs {
  nn::Tensor fine, passthrough, features, raw;  // raw [N, c, c, anchors, 5 + classes]
};

/// conv (no bias) -> BN (scale, eps 1e-5) -> leaky ReLU .1, symmetric padding.
inline nn::Tensor DarknetConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
                              std::int64_t out, std::int64_t kernel, bool training, double momentum, bool update_stats,
                              std::int64_t stride = 1) {
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.momentum     = momentum;
  norm.epsilon      = 1e-5;
  norm.update_stats = update_stats;
  return nn::leaky_relu(resnet::ResNetConv(p, state, name, x, out, kernel, norm, stride, false), .1);
}

inline YoloV2Outputs YoloV2(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image, const YoloV2Config& cfg = {},
                            bool training = false) {
  if (cfg.classes < 1 || cfg.anchors < 1 || cfg.width_divisor < 1 || 32 % cfg.width_divisor || !(cfg.momentum >= 0 && cfg.momentum < 1))
    throw std::invalid_argument("invalid YOLO v2 configuration");
  const std::vector<std::vector<std::pair<int, int>>> stages = {
      {{32, 3}}, {{64, 3}}, {{128, 3}, {64, 1}, {128, 3}}, {{256, 3}, {128, 1}, {256, 3}},
      {{512, 3}, {256, 1}, {512, 3}, {256, 1}, {512, 3}}, {{1024, 3}, {512, 1}, {1024, 3}, {512, 1}, {1024, 3}}};
  YoloV2Outputs result;
  nn::Tensor    h = image;
  std::int64_t  in = 0;
  int           index = 0;
  auto conv = [&](const std::string& name, const nn::Tensor& x, std::int64_t out, std::int64_t kernel) {
    return DarknetConv(p, state, name, x, out, kernel, training, cfg.momentum, cfg.update_stats);
  };
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    for (const auto& [channels, kernel] : stages[stage]) {
      in = channels / cfg.width_divisor;
      h  = conv("backbone/conv" + std::to_string(++index), h, in, kernel);
    }
    if (stage == 4) result.fine = h;
    if (stage < 5 && (!cfg.compact || stage >= 2)) h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  h                  = conv("extra1", h, in, 3);
  h                  = conv("extra2", h, in, 3);
  const auto route   = conv("route", result.fine, 64 / cfg.width_divisor, 1);
  result.passthrough = nn::space_to_depth(route, 2);
  h                  = conv("fusion", nn::concat({h, result.passthrough}, 3), in, 3);
  result.features    = h;
  nn::ConvLayerOptions linear;
  linear.init           = nn::init::he_uniform();
  const auto prediction = nn::conv2d(p, "prediction", h, cfg.anchors * (5 + cfg.classes), 1, linear);
  const auto& s         = prediction.shape();
  result.raw            = nn::reshape(prediction, {s[0], s[1], s[2], cfg.anchors, 5 + cfg.classes});
  return result;
}

// ---- priors, decoding and targets ----------------------------------------------------

using Dimension = std::array<float, 2>;  // height, width
inline float DimensionIoU(const Dimension& a, const Dimension& b) {
  const float intersection = std::min(a[0], b[0]) * std::min(a[1], b[1]);
  return intersection / (a[0] * a[1] + b[0] * b[1] - intersection);
}
/// IoU k-means (1 - IoU distance), farthest-point seeded from the first sample; sorted by area.
inline std::vector<Dimension> ClusterDimensions(const std::vector<Dimension>& samples, int count) {
  if (count < 1 || samples.size() < std::size_t(count)) throw std::invalid_argument("invalid dimension-cluster count");
  for (const auto& s : samples)
    if (!(s[0] > 0 && s[1] > 0) || !std::isfinite(s[0] + s[1])) throw std::invalid_argument("dimension samples must be finite and positive");
  std::vector<Dimension> centres = {samples[0]};
  while (int(centres.size()) < count) {
    std::size_t farthest = 0;
    float       distance = -1;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      float nearest = 1;
      for (const auto& c : centres) nearest = std::min(nearest, 1 - DimensionIoU(samples[i], c));
      if (nearest > distance) {
        distance = nearest;
        farthest = i;
      }
    }
    centres.push_back(samples[farthest]);
  }
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::vector<std::array<double, 2>> sum(std::size_t(count), {0, 0});
    std::vector<int>                   used(std::size_t(count), 0);
    for (const auto& sample : samples) {
      int best = 0;
      for (int k = 1; k < count; ++k)
        if (DimensionIoU(sample, centres[std::size_t(k)]) > DimensionIoU(sample, centres[std::size_t(best)])) best = k;
      sum[std::size_t(best)][0] += sample[0];
      sum[std::size_t(best)][1] += sample[1];
      ++used[std::size_t(best)];
    }
    double change = 0;
    for (std::size_t k = 0; k < std::size_t(count); ++k)
      if (used[k])
        for (std::size_t d = 0; d < 2; ++d) {
          const float next = float(sum[k][d] / used[k]);
          change           = std::max(change, std::abs(double(next - centres[k][d])));
          centres[k][d]    = next;
        }
    if (change < 1e-5) break;
  }
  std::sort(centres.begin(), centres.end(), [](const auto& a, const auto& b) { return a[0] * a[1] < b[0] * b[1]; });
  return centres;
}
inline Box DecodeYoloV2(const float* raw, int row, int col, int cells, const Dimension& prior, int height, int width) {
  auto        sigmoid = [](float v) { return float(1. / (1. + std::exp(-double(v)))); };
  const float y = (float(row) + sigmoid(raw[0])) / float(cells) * float(height), x = (float(col) + sigmoid(raw[1])) / float(cells) * float(width);
  const float h = prior[0] * std::exp(std::clamp(raw[2], -10.f, 10.f)) / float(cells) * float(height);
  const float w = prior[1] * std::exp(std::clamp(raw[3], -10.f, 10.f)) / float(cells) * float(width);
  return {y - h / 2, x - w / 2, y + h / 2, x + w / 2};
}

struct YoloV2Targets {
  std::vector<float> geometry, positive, confidence, weights;  // [N,c,c,A,4], [N,c,c,A] x3
  std::vector<int>   labels;                                   // [N,c,c,A]
  int                collisions = 0;
};
/// Targets from detached predictions `raw` (host values, shape [N, c, c, A, channels]).
inline YoloV2Targets MakeYoloV2Targets(const std::vector<float>& raw, const nn::Shape& shape, const std::vector<Dimension>& priors,
                                       const DetectionScenes& scenes, const std::vector<int>& rows, bool coarse) {
  if (shape.size() != 5) throw std::invalid_argument("YOLO predictions must have rank five");
  const std::int64_t batch = std::int64_t(rows.size());
  const int          cells = int(shape[1]), anchors = int(priors.size()), channels = int(shape[4]);
  if (shape[0] != batch || shape[2] != cells || shape[3] != anchors) throw std::invalid_argument("YOLO target/prediction dimensions do not match");
  const std::size_t slots = std::size_t(batch) * std::size_t(cells * cells * anchors);
  YoloV2Targets     t;
  t.geometry.assign(slots * 4, 0);
  t.labels.assign(slots, 0);
  t.positive.assign(slots, 0);
  t.confidence.assign(slots, 0);
  t.weights.assign(slots, .5f);
  auto at = [&](std::int64_t i, int r, int c, int a) { return std::size_t(((i * cells + r) * cells + c) * anchors + a); };
  for (std::int64_t i = 0; i < batch; ++i) {
    const auto& truth = scenes.boxes[std::size_t(rows[std::size_t(i)])];
    for (int r = 0; r < cells; ++r)
      for (int c = 0; c < cells; ++c)
        for (int a = 0; a < anchors; ++a) {
          const auto box = DecodeYoloV2(raw.data() + at(i, r, c, a) * std::size_t(channels), r, c, cells, priors[std::size_t(a)], scenes.height,
                                        scenes.width);
          float overlap = 0;
          for (const auto& target : truth) overlap = std::max(overlap, detector::BoxIoU(box, target));
          if (overlap > .6f) t.weights[at(i, r, c, a)] = 0;
        }
    std::vector<float> areas(std::size_t(cells * cells * anchors), 0);
    for (std::size_t g = 0; g < truth.size(); ++g) {
      const auto&     b = truth[g];
      const float     y = (b[0] + b[2]) * .5f / float(scenes.height) * float(cells), x = (b[1] + b[3]) * .5f / float(scenes.width) * float(cells);
      const Dimension size = {(b[2] - b[0]) / float(scenes.height) * float(cells), (b[3] - b[1]) / float(scenes.width) * float(cells)};
      const int       r = std::clamp(int(y), 0, cells - 1), c = std::clamp(int(x), 0, cells - 1);
      int             a = 0;
      for (int k = 1; k < anchors; ++k)
        if (DimensionIoU(size, priors[std::size_t(k)]) > DimensionIoU(size, priors[std::size_t(a)])) a = k;
      const std::size_t slot = std::size_t((r * cells + c) * anchors + a);
      const float       area = size[0] * size[1];
      if (areas[slot] > 0) {
        ++t.collisions;
        if (area <= areas[slot]) continue;
      }
      areas[slot]        = area;
      const auto  k      = at(i, r, c, a);
      const auto  current = DecodeYoloV2(raw.data() + k * std::size_t(channels), r, c, cells, priors[std::size_t(a)], scenes.height, scenes.width);
      const float coordinates[] = {y - float(r), x - float(c), std::log(size[0] / priors[std::size_t(a)][0]), std::log(size[1] / priors[std::size_t(a)][1])};
      for (std::size_t d = 0; d < 4; ++d) t.geometry[k * 4 + d] = coordinates[d];
      const int label = scenes.labels[std::size_t(rows[std::size_t(i)])][g];
      t.labels[k]     = coarse ? label % 2 : label;
      t.positive[k] = t.weights[k] = 1;
      t.confidence[k]              = detector::BoxIoU(current, b);
    }
  }
  return t;
}

/// Class probabilities [N, c, c, A, classes]: softmax, or WordTree marginals.
inline nn::Tensor YoloV2Probabilities(const nn::Tensor& raw, int classes, const wordtree::WordTree* tree) {
  const auto& s      = raw.shape();
  const auto  logits = nn::reshape(nn::slice(raw, {0, 0, 0, 0, 5}, {s[0], s[1], s[2], s[3], classes}), {-1, classes});
  const auto  p      = tree ? wordtree::WordTreeProbabilities(logits, *tree).marginal : nn::softmax(logits, -1);
  return nn::reshape(p, {s[0], s[1], s[2], s[3], classes});
}

struct YoloV2Weights {
  double detection = 1, classification = 0;
  std::vector<int> classification_labels;  // one per image (WordTree node)
};
inline nn::Tensor YoloV2Loss(const nn::Tensor& raw, const YoloV2Targets& t, int classes, const YoloV2Weights& w,
                             const wordtree::WordTree* tree = nullptr) {
  const auto&        s       = raw.shape();
  const std::int64_t n = s[0], gh = s[1], gw = s[2], anchors = s[3];
  if (tree && tree->nodes() != classes) throw std::invalid_argument("YOLO class count must equal WordTree nodes");
  const nn::Shape    grid    = {n, gh, gw, anchors};
  const auto         encoded = nn::concat({nn::sigmoid(nn::slice(raw, {0, 0, 0, 0, 0}, {n, gh, gw, anchors, 2})),
                                           nn::slice(raw, {0, 0, 0, 0, 2}, {n, gh, gw, anchors, 2})}, 4);
  const auto objectness   = nn::squeeze(nn::sigmoid(nn::slice(raw, {0, 0, 0, 0, 4}, {n, gh, gw, anchors, 1})), {4});
  const auto class_logits = nn::reshape(nn::slice(raw, {0, 0, 0, 0, 5}, {n, gh, gw, anchors, classes}), {-1, classes});
  nn::Tensor class_loss;
  if (tree) {
    class_loss = wordtree::WordTreeLoss(wordtree::WordTreeProbabilities(class_logits, *tree), t.labels, *tree);
  } else {
    const auto one_hot = nn::one_hot(nn::from_vector(std::vector<std::int32_t>(t.labels.begin(), t.labels.end())), classes);
    class_loss         = -nn::sum(nn::log_softmax(class_logits, -1) * one_hot, {1});
  }
  const auto positive     = nn::from_vector(t.positive, grid);
  const auto class_total  = nn::sum(class_loss * nn::reshape(positive, {-1}));
  const auto coordinates  = nn::sum(nn::sum(nn::square(encoded - nn::from_vector(t.geometry, {n, gh, gw, anchors, 4})), {4}) * positive);
  const auto object_total = nn::sum(nn::square(objectness - nn::from_vector(t.confidence, grid)) * nn::from_vector(t.weights, grid));
  nn::Tensor loss = (coordinates * 5.0 + object_total + class_total) / nn::maximum(nn::sum(positive), nn::scalar(1.0)) * w.detection;
  if (tree && w.classification != 0) {
    if (std::int64_t(w.classification_labels.size()) != n) throw std::invalid_argument("one classification label per image");
    const std::int64_t per_image = gh * gw * anchors;
    std::vector<std::int32_t> flat;
    {
      nn::NoGradGuard guard;
      const auto      marginal = wordtree::WordTreeProbabilities(class_logits, *tree).marginal.to_vector<float>();
      const auto      object   = objectness.to_vector<float>();
      for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t best = 0;
        float        best_score = -1;
        for (std::int64_t k = 0; k < per_image; ++k) {
          const float score = marginal[std::size_t((i * per_image + k) * classes + w.classification_labels[std::size_t(i)])] *
                              object[std::size_t(i * per_image + k)];
          if (score > best_score) {
            best_score = score;
            best       = k;
          }
        }
        flat.push_back(std::int32_t(i * per_image + best));
      }
    }
    const auto indices        = nn::from_vector(flat);
    const auto selected       = wordtree::WordTreeProbabilities(nn::gather(class_logits, indices, 0), *tree);
    const auto classification = wordtree::WordTreeLoss(selected, w.classification_labels, *tree);
    const auto weak           = nn::square(nn::relu(.3 - nn::gather(nn::reshape(objectness, {-1}), indices, 0)));
    loss                      = loss + nn::mean(classification + weak) * w.classification;
  }
  return loss;
}

/// Detections of rows `chunk` from raw predictions and class probabilities (host); no NMS.
inline void DecodeYoloV2Detections(const std::vector<float>& raw, const nn::Shape& shape, const std::vector<float>& probabilities, int classes,
                                   const std::vector<Dimension>& priors, const DetectionScenes& scenes, const std::vector<int>& chunk,
                                   bool hierarchy, std::vector<detector::Detection>& detections) {
  const int cells = int(shape[1]), anchors = int(priors.size()), channels = int(shape[4]);
  for (std::size_t i = 0; i < chunk.size(); ++i)
    for (int r = 0; r < cells; ++r)
      for (int c = 0; c < cells; ++c)
        for (int a = 0; a < anchors; ++a) {
          const std::size_t slot = ((i * std::size_t(cells) + std::size_t(r)) * std::size_t(cells) + std::size_t(c)) * std::size_t(anchors) + std::size_t(a);
          const float*      v    = raw.data() + slot * std::size_t(channels);
          const auto        b    = detector::ClipBox(DecodeYoloV2(v, r, c, cells, priors[std::size_t(a)], scenes.height, scenes.width),
                                                     float(scenes.height), float(scenes.width));
          if (!(detector::BoxArea(b) > 0)) continue;
          const float  objectness  = float(1. / (1. + std::exp(-double(v[4]))));
          const float* probability = probabilities.data() + slot * std::size_t(classes);
          for (int label = 0; label < 10; ++label) {
            const float score = objectness * probability[hierarchy ? label + 2 : label];
            if (score > .01f) detections.push_back({chunk[i], label, score, b});
          }
        }
}
/// Inference in chunks of 32 with `forward(images)` returning raw predictions; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection> DetectYoloV2(const Forward& forward, const std::vector<Dimension>& priors, const DetectionScenes& scenes,
                                              const std::vector<int>& rows, int classes, const wordtree::WordTree* tree) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const nn::Tensor       raw = forward(detector::SceneImages(scenes, chunk));
    DecodeYoloV2Detections(raw.to_vector<float>(), raw.shape(), YoloV2Probabilities(raw, classes, tree).to_vector<float>(), classes, priors,
                           scenes, chunk, tree != nullptr, detections);
  }
  return detector::ClassNms(detections, .45f);
}

}  // namespace yolo
