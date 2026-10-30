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

using nn::models::RetinaConv;
inline nn::Tensor PadSpatial(const nn::Tensor &x, std::int64_t pad) {
  return nn::models::RetinaPadSpatial(x, pad);
}
using nn::models::RetinaLoss;
using nn::models::RetinaNet;
using nn::models::RetinaNetConfig;
using nn::models::RetinaNetOutputs;
using nn::models::RetinaPyramid;
using nn::models::RetinaTargets;
using nn::models::SigmoidFocalLoss;
inline std::vector<Box> RetinaAnchors(const std::vector<int> &cells, float first_size = 32) {
  return nn::boxes_to_vector(nn::models::RetinaAnchors(cells, first_size));
}
inline RetinaTargets MakeRetinaTargets(const std::vector<Box> &anchors,
                                       const DetectionScenes &scenes,
                                       const std::vector<int> &rows, int classes) {
  nn::models::TensorDetectionScenes data;
  const auto N = std::int64_t(rows.size());
  if (!N)
    return {nn::zeros({0, std::int64_t(anchors.size()), classes}),
            nn::zeros({0, std::int64_t(anchors.size())}),
            nn::zeros({0, std::int64_t(anchors.size())}),
            nn::zeros({0, std::int64_t(anchors.size()), 4})};
  data.images = nn::zeros({N, scenes.height, scenes.width, 1});
  std::vector<std::int64_t> selected;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto row = rows[i];
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("Retina row outside scenes");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
    selected.push_back(std::int64_t(i));
  }
  return nn::models::MakeRetinaTargets(nn::boxes_from_vector(anchors), data, selected,
                                       classes);
}
inline std::vector<detector::Detection>
DecodeRetina(const nn::Tensor &probabilities, const nn::Tensor &deltas,
             const std::vector<Box> &anchors, const DetectionScenes &scenes,
             const std::vector<int> &rows, int classes) {
  if (probabilities.dim() != 3 || probabilities.shape()[2] != classes)
    throw std::invalid_argument("Retina class count mismatch");
  const auto decoded = nn::models::DecodeRetinaDetections(
      probabilities, deltas, nn::boxes_from_vector(anchors).to(probabilities.dtype()),
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width, false);
  std::vector<detector::Detection> out;
  for (const auto &d : nn::models::DetectionEvaluationRecords(decoded))
    out.push_back({d.image, d.label, d.score, d.box});
  return out;
}

/// Inference in chunks of 32 scenes with `forward(images)` returning RetinaNetOutputs; class-wise NMS .5.
template <typename Forward>
std::vector<detector::Detection>
DetectRetina(const Forward &forward, const std::vector<Box> &anchors,
             const DetectionScenes &scenes, const std::vector<int> &rows, int classes) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const RetinaNetOutputs net = forward(detector::SceneImages(scenes, chunk));
    const auto found =
        DecodeRetina(net.probabilities, net.deltas, anchors, scenes, chunk, classes);
    result.insert(result.end(), found.begin(), found.end());
  }
  return detector::ClassNms(result, .5f);
}

} // namespace retinanet
