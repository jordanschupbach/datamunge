// FCOS: anchor-free distances and centerness on a ResNet P3-P7 pyramid, with
// point assignment, centerness, focal/IoU losses and decoding (port of
// tflow's common/nn/fcos.h and common/ml/fcos.h).
// https://arxiv.org/abs/1904.01355
// https://github.com/tianzhi0549/FCOS
//
// Model: shared classification/regression towers each have four
// 3x3/GN/ReLU layers. Each level learns one scalar before exponentiating box
// distances (l,t,r,b). Centerness comes from the classification tower, as in
// the original model. Full defaults: ResNet-50, width256, GN32, 80 classes.
// Compact demonstrations use ResNet-18/8, width32, GN8 and exp distances
// times stride; this explicit scale adaptation keeps compact scratch training
// well conditioned.
//
// Targets/loss: points are stride/2 + integer*stride. Positive points lie
// strictly inside boxes and within that pyramid level's regression range.
// Optional radius1.5 centre sampling follows the improved reference; overlaps
// choose smallest area. Centerness is the square root of
// min(l,r)/max(l,r)*min(t,b)/max(t,b). Focal classification and positive
// centerness BCE normalize by positive count. Positive -log IoU uses +1 area
// smoothing, weights by centerness and divides by total centerness.
// Inference ranks sqrt(class probability * centerness), followed by
// class-wise NMS.
#pragma once

#include "common/retinanet.hpp"

namespace fcos {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

struct FcosConfig {
  int  classes = 80, image_size = 640, backbone_depth = 50, width_divisor = 1, pyramid_width = 256, groups = 32;
  bool grayscale = false, stride_scaled_distances = false;
  bool update_stats = true;
};
struct FcosOutputs {
  std::vector<nn::Tensor> features;
  std::vector<int>        cells;
  nn::Tensor              logits, distances, centerness;  // [N, P, C], [N, P, 4], [N, P]
  int                     points = 0;
};
inline FcosOutputs Fcos(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images,
                        const FcosConfig& cfg = {}, bool training = false) {
  if (cfg.classes < 1 || cfg.image_size < 32 || cfg.image_size % 32 || cfg.groups < 1 || cfg.pyramid_width % cfg.groups)
    throw std::invalid_argument("invalid FCOS configuration");
  FcosOutputs result;
  result.features = retinanet::RetinaPyramid(p, state, images, cfg.backbone_depth, cfg.width_divisor, cfg.pyramid_width,
                                             cfg.grayscale, training, cfg.update_stats);
  std::vector<nn::Tensor> logits, distances, centerness;
  for (int level = 0; level < 5; ++level) {
    const int stride = 8 << level, cells = (cfg.image_size + stride - 1) / stride, count = cells * cells;
    if (result.features[std::size_t(level)].shape()[1] != cells)
      throw std::invalid_argument("FCOS image_size does not match the input images");
    result.cells.push_back(cells);
    result.points += count;
    nn::Tensor classification = result.features[std::size_t(level)], regression = classification;
    for (int i = 0; i < 4; ++i) {
      const auto name = std::to_string(i + 1);
      classification  = nn::relu(nn::group_norm(
          p, "classification/gn" + name,
          retinanet::RetinaConv(p, "classification/conv" + name, classification, cfg.pyramid_width, false), cfg.groups));
      regression = nn::relu(nn::group_norm(p, "regression/gn" + name,
                                           retinanet::RetinaConv(p, "regression/conv" + name, regression, cfg.pyramid_width, false),
                                           cfg.groups));
    }
    logits.push_back(nn::reshape(
        retinanet::RetinaConv(p, "classification/output", classification, cfg.classes, false, std::log(.01 / .99)),
        {-1, count, cfg.classes}));
    centerness.push_back(nn::reshape(retinanet::RetinaConv(p, "centerness", classification, 1, false), {-1, count}));
    const auto raw   = retinanet::RetinaConv(p, "regression/output", regression, 4, false);
    const auto scale = p.get("scale" + std::to_string(level), {}, nn::init::ones());
    // Limit the exponent only far beyond useful image distances, preventing overflow.
    const auto positive = nn::exp(nn::minimum(raw * scale, nn::scalar(20.0)));
    distances.push_back(nn::reshape(positive * (cfg.stride_scaled_distances ? double(stride) : 1.0), {-1, count, 4}));
  }
  result.logits     = nn::concat(logits, 1);
  result.distances  = nn::concat(distances, 1);
  result.centerness = nn::concat(centerness, 1);
  return result;
}

// ---- points, targets, loss and decoding --------------------------------------------

struct FcosPoint {
  float y, x, stride, minimum, maximum;
};
inline std::vector<FcosPoint> FcosPoints(const std::vector<int>& cells, float range_scale = 1.f) {
  if (!(range_scale > 0) || !std::isfinite(range_scale) || cells.size() != 5)
    throw std::invalid_argument("invalid FCOS pyramid/range scale");
  std::vector<FcosPoint> result;
  for (int level = 0; level < 5; ++level) {
    const float stride  = float(8 << level), minimum = level ? std::ldexp(32.f * range_scale, level) : 0,
                maximum = level < 4 ? std::ldexp(64.f * range_scale, level) : INFINITY;
    for (int r = 0; r < cells[std::size_t(level)]; ++r)
      for (int c = 0; c < cells[std::size_t(level)]; ++c)
        result.push_back({(float(r) + .5f) * stride, (float(c) + .5f) * stride, stride, minimum, maximum});
  }
  return result;
}
inline float FcosCenterness(const std::array<float, 4>& d) {
  if (*std::min_element(d.begin(), d.end()) <= 0) return 0;
  return std::sqrt(std::min(d[0], d[2]) / std::max(d[0], d[2]) * std::min(d[1], d[3]) / std::max(d[1], d[3]));
}
/// -log((I + 1) / (U + 1)) of (l, t, r, b) distance boxes [N, P, 4] -> [N, P].
inline nn::Tensor FcosIouLoss(const nn::Tensor& prediction, const nn::Tensor& target) {
  const auto a = nn::unstack(prediction, 2), b = nn::unstack(target, 2);
  const auto area_a       = (a[0] + a[2]) * (a[1] + a[3]);
  const auto area_b       = (b[0] + b[2]) * (b[1] + b[3]);
  const auto intersection = (nn::minimum(a[0], b[0]) + nn::minimum(a[2], b[2])) * (nn::minimum(a[1], b[1]) + nn::minimum(a[3], b[3]));
  return -nn::log((intersection + 1.0) / (area_a + area_b - intersection + 1.0));
}
struct FcosTargets {
  nn::Tensor classes, distances, positive, centerness;  // [N, P, C], [N, P, 4], [N, P], [N, P]
};
inline nn::Tensor FcosLoss(const nn::Tensor& logits, const nn::Tensor& distances, const nn::Tensor& centerness,
                           const FcosTargets& t) {
  const auto count          = nn::maximum(nn::sum(t.positive), nn::scalar(1.0));
  const auto classification = nn::sum(retinanet::SigmoidFocalLoss(logits, t.classes)) / count;
  const auto center_bce     = nn::softplus(centerness) - centerness * t.centerness;
  const auto center         = nn::sum(center_bce * t.positive) / count;
  const auto weights        = t.centerness * t.positive;
  const auto regression     = nn::sum(FcosIouLoss(distances, t.distances) * weights) / nn::maximum(nn::sum(weights), nn::scalar(1e-6));
  return classification + (center + regression);
}
inline nn::Tensor FcosLoss(const FcosOutputs& net, const FcosTargets& t) { return FcosLoss(net.logits, net.distances, net.centerness, t); }

/// Host arrays of the targets, for fixtures.
struct FcosTargetArrays {
  std::vector<float> classes, distances, positive, centerness;
};
inline FcosTargetArrays MakeFcosTargetArrays(const std::vector<FcosPoint>& points, const DetectionScenes& scenes,
                                             const std::vector<int>& rows, int classes, float center_radius = 1.5f) {
  const std::size_t batch = rows.size(), count = points.size();
  FcosTargetArrays  t{std::vector<float>(batch * count * std::size_t(classes), 0), std::vector<float>(batch * count * 4, 0),
                     std::vector<float>(batch * count, 0), std::vector<float>(batch * count, 0)};
  for (std::size_t i = 0; i < batch; ++i)
    for (std::size_t a = 0; a < count; ++a) {
      const auto&          point = points[a];
      int                  best  = -1;
      float                area  = INFINITY;
      std::array<float, 4> distance{};
      const auto&          truth = scenes.boxes[std::size_t(rows[i])];
      for (std::size_t g = 0; g < truth.size(); ++g) {
        const auto&                b = truth[g];
        const std::array<float, 4> d = {point.x - b[1], point.y - b[0], b[3] - point.x, b[2] - point.y};
        if (*std::min_element(d.begin(), d.end()) <= 0) continue;
        const float largest = *std::max_element(d.begin(), d.end());
        if (largest < point.minimum || largest > point.maximum) continue;
        if (center_radius > 0 && (std::abs(point.x - (b[1] + b[3]) / 2) >= center_radius * point.stride ||
                                  std::abs(point.y - (b[0] + b[2]) / 2) >= center_radius * point.stride))
          continue;
        const float candidate = detector::BoxArea(b);
        if (candidate < area) {
          best     = int(g);
          area     = candidate;
          distance = d;
        }
      }
      if (best < 0) continue;
      const int label = scenes.labels[std::size_t(rows[i])][std::size_t(best)];
      if (label < 0 || label >= classes) throw std::invalid_argument("FCOS target outside class range");
      const std::size_t at = i * count + a;
      t.classes[at * std::size_t(classes) + std::size_t(label)] = t.positive[at] = 1;
      t.centerness[at] = FcosCenterness(distance);
      for (std::size_t d = 0; d < 4; ++d) t.distances[at * 4 + d] = distance[d];
    }
  return t;
}
inline FcosTargets MakeFcosTargets(const std::vector<FcosPoint>& points, const DetectionScenes& scenes,
                                   const std::vector<int>& rows, int classes, float center_radius = 1.5f) {
  const auto t = MakeFcosTargetArrays(points, scenes, rows, classes, center_radius);
  const auto N = std::int64_t(rows.size()), P = std::int64_t(points.size());
  return {nn::from_vector(t.classes, {N, P, classes}), nn::from_vector(t.distances, {N, P, 4}),
          nn::from_vector(t.positive, {N, P}), nn::from_vector(t.centerness, {N, P})};
}

/// Best 1000 candidates per image by sqrt(probability * centerness) (probability > .01).
inline std::vector<detector::Detection> DecodeFcos(const FcosOutputs& net, const std::vector<FcosPoint>& points,
                                                   const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  const auto L = net.logits.to_vector<float>(), D = net.distances.to_vector<float>(), C = net.centerness.to_vector<float>();
  const std::size_t                count = points.size();
  std::vector<detector::Detection> result;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::vector<detector::Detection> candidates;
    for (std::size_t a = 0; a < count; ++a)
      for (int c = 0; c < classes; ++c) {
        const float probability = 1 / (1 + std::exp(-L[(i * count + a) * std::size_t(classes) + std::size_t(c)]));
        if (probability <= .01f) continue;
        const float  score = std::sqrt(probability / (1 + std::exp(-C[i * count + a])));
        const auto&  p     = points[a];
        const float* d     = D.data() + (i * count + a) * 4;
        const Box    b = detector::ClipBox({p.y - d[1], p.x - d[0], p.y + d[3], p.x + d[2]}, float(scenes.height), float(scenes.width));
        if (detector::BoxArea(b) > 0) candidates.push_back({rows[i], c, score, b});
      }
    const std::size_t keep = std::min<std::size_t>(1000, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + std::ptrdiff_t(keep), candidates.end(),
                      [](const auto& x, const auto& y) { return x.score > y.score; });
    result.insert(result.end(), candidates.begin(), candidates.begin() + std::ptrdiff_t(keep));
  }
  return result;
}
inline std::vector<detector::Detection> DetectFcos(nn::ParameterStore& p, nn::ParameterStore& state, const FcosConfig& cfg,
                                                   const std::vector<FcosPoint>& points, const DetectionScenes& scenes,
                                                   const std::vector<int>& rows, int classes) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto             net   = Fcos(p, state, detector::SceneImages(scenes, chunk), cfg, false);
    const auto             found = DecodeFcos(net, points, scenes, chunk, classes);
    result.insert(result.end(), found.begin(), found.end());
  }
  return detector::ClassNms(result, .6f, 100);
}

}  // namespace fcos
