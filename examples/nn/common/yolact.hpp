#pragma once
// Compatibility containers;canonical tensor library owns all numerical work.
#include "common/detection_scenes.hpp"
#include "common/instance_segmentation.hpp"
#include "common/resnet.hpp"
#include "datamunge/nn/yolact_models.hpp"
#include <random>
namespace yolact {
namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;
using nn::models::Yolact;
using nn::models::YolactCells;
using nn::models::YolactConfig;
using nn::models::YolactOutputs;
inline nn::Tensor LinearMasks(const nn::Tensor &prototypes, const nn::Tensor &coefficients) {
  return nn::models::LinearInstanceMasks(prototypes, coefficients);
}
inline nn::Tensor CroppedMaskLoss(const nn::Tensor &logits, const nn::Tensor &targets,
                                  const nn::Tensor &crop) {
  return nn::models::CroppedInstanceMaskLoss(logits, targets, crop);
}
inline std::vector<int> YolactMatching(const std::vector<Box> &anchors,
                                       const std::vector<Box> &truth) {
  return nn::models::YolactMatching(nn::boxes_from_vector(anchors),
                                    nn::boxes_from_vector(truth))
      .to_vector<int>();
}
inline std::vector<Box> YolactAnchors(const YolactOutputs &net, int size, bool compact) {
  return nn::boxes_to_vector(nn::models::YolactAnchors(net, size, compact));
}
struct YolactTargets {
  int batch = 0, anchors = 0, side = 0, cells = 0, classes = 0;
  std::vector<std::int32_t> labels;
  std::vector<float> deltas;
  std::vector<std::pair<int, int>> pairs;
  std::vector<float> mask_targets, crop, semantic;
};
struct YolactLoss {
  nn::Tensor loss, mask_loss;
  std::vector<char> negatives;
};
inline nn::models::TensorYolactTargets TensorTargets(const YolactTargets &t) {
  std::vector<std::int64_t> pairs;
  for (const auto &[image, anchor] : t.pairs) {
    pairs.push_back(image);
    pairs.push_back(anchor);
  }
  const auto M = std::int64_t(t.pairs.size());
  return {nn::from_vector(std::vector<std::int64_t>(t.labels.begin(), t.labels.end()),
                          {t.batch, t.anchors}),
          nn::from_vector(t.deltas, {t.batch, t.anchors, 4}),
          nn::from_vector(pairs, {M, 2}),
          nn::from_vector(t.mask_targets, {M, t.side, t.side}),
          nn::from_vector(t.crop, {M, t.side, t.side}),
          nn::from_vector(t.semantic, {t.batch, t.cells, t.cells, t.classes})};
}
inline YolactLoss BuildYolactLoss(const YolactOutputs &net, const YolactTargets &target) {
  const auto value = nn::models::BuildTensorYolactLoss(net, TensorTargets(target));
  const auto flags = value.negatives.to_vector<int>();
  return {value.loss, value.mask_loss, std::vector<char>(flags.begin(), flags.end())};
}
inline std::vector<int> FastNms(const std::vector<Detection> &detections,
                                float threshold = .5f, int top_k = 200, int limit = 100) {
  std::vector<Box> boxes;
  std::vector<float> scores;
  std::vector<std::int64_t> labels, images;
  for (const auto &d : detections) {
    boxes.push_back(d.box);
    scores.push_back(d.score);
    labels.push_back(d.label);
    images.push_back(d.image);
  }
  return nn::models::TensorFastNms({nn::boxes_from_vector(boxes), nn::from_vector(scores),
                                    nn::from_vector(labels), nn::from_vector(images)},
                                   threshold, top_k, limit)
      .to_vector<int>();
}
inline YolactTargets MakeYolactTargets(const YolactOutputs &net,
                                       const std::vector<Box> &anchors,
                                       const instance::InstanceScenes &scenes,
                                       const std::vector<int> &rows, int classes,
                                       std::mt19937 &random) {
  nn::Generator rng(random());
  const auto target = nn::models::MakeTensorYolactTargets(
      net, nn::boxes_from_vector(anchors), instance::TensorAnnotations(scenes),
      std::vector<std::int64_t>(rows.begin(), rows.end()), classes, rng);
  YolactTargets out;
  out.batch = int(rows.size());
  out.anchors = int(anchors.size());
  out.side = net.proto_side;
  out.cells = net.cells[0];
  out.classes = classes;
  out.labels = target.labels.to_vector<std::int32_t>();
  out.deltas = target.deltas.to_vector<float>();
  const auto pairs = target.pairs.to_vector<int>();
  for (std::size_t k = 0; k < pairs.size(); k += 2)
    out.pairs.push_back({pairs[k], pairs[k + 1]});
  out.mask_targets = target.mask_targets.to_vector<float>();
  out.crop = target.crop.to_vector<float>();
  out.semantic = target.semantic.to_vector<float>();
  return out;
}
inline std::vector<instance::MaskDetection>
DecodeYolact(const YolactOutputs &net, const std::vector<Box> &anchors,
             const instance::InstanceScenes &scenes, const std::vector<int> &chunk,
             int classes) {
  if (net.probabilities.dim() != 3 || net.probabilities.shape()[2] != classes + 1)
    throw std::invalid_argument("YOLACT decode class mismatch");
  const auto output = nn::models::DecodeTensorYolact(
      net, nn::boxes_from_vector(anchors),
      std::vector<std::int64_t>(chunk.begin(), chunk.end()), scenes.height, scenes.width);
  const auto records = nn::models::DetectionEvaluationRecords(output.objects);
  std::vector<instance::MaskDetection> result;
  for (std::size_t k = 0; k < records.size(); ++k) {
    const auto &d = records[k];
    const auto mask = nn::slice_axis(output.masks, 0, k, k + 1).to_vector<int>();
    result.push_back(
        {{d.image, d.label, d.score, d.box}, instance::BinaryMask(mask.begin(), mask.end())});
  }
  return result;
}
template <typename InferFn>
std::vector<instance::MaskDetection> DetectYolact(const InferFn &infer,
                                                  const std::vector<Box> &anchors,
                                                  const instance::InstanceScenes &scenes,
                                                  const std::vector<int> &rows, int classes) {
  nn::NoGradGuard guard;
  std::vector<instance::MaskDetection> result;
  for (std::size_t start = 0; start < rows.size(); start += 8) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 8)));
    const auto part = DecodeYolact(infer(detector::SceneImages(scenes, chunk)), anchors,
                                   scenes, chunk, classes);
    result.insert(result.end(), part.begin(), part.end());
  }
  return result;
}

} // namespace yolact
