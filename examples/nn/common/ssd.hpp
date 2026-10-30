// SSD: VGG-16 SSD300 and the compact digit-scene variant (port of tflow's
// common/nn/ssd.h), default boxes, the multibox loss with per-image 3:1 hard
// negative mining, targets and batched decoding (port of common/ml/ssd.h).
// https://arxiv.org/abs/1512.02325
// https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
//
// Full models use conv4_3 with learned L2 rescaling (initial scale 20),
// atrous fc6 (dilation 6), fc7 and four extra blocks. The six grids are
// 38/19/10/5/3/1 for a 300-pixel image, with 4/6/6/6/4/4 boxes per cell.
// The compact option uses a small 64-pixel grayscale backbone and grids
// 16/8/4/2/1/1 (an explicit training-demonstration adaptation). Kernels use
// He-uniform initialization; no pretrained weights are loaded. fc6 uses the
// library's dilated convolution (tflow expands the kernel with zeros, which
// is the same operation).
//
// Background is the last class. Matching (detector::MatchSsd) first greedily
// gives every target a distinct default box, then assigns the remaining boxes
// with IoU >= 0.5. Hard negatives are ranked per image by softmax
// cross-entropy (positives last, ties by lower index as TF TopK); at most
// three per positive are kept. The loss is summed over the batch and divided
// by the total positives (zero for an all-background batch). The selection
// delegates to tensor argsort on detached cross-entropies; the discrete
// selection is not differentiated, like TF TopK. Deltas use variances 0.1/0.1/0.2/0.2;
// inference boxes are clipped before class-wise NMS (.45).
#pragma once

#include "common/detection_scenes.hpp"
#include "common/vgg.hpp"

namespace ssd {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

inline std::vector<Box> SsdDefaultBoxes(const std::vector<int> &cells,
                                        const std::vector<float> &sizes,
                                        const std::vector<float> &steps,
                                        const std::vector<int> &per_cell) {
  return nn::boxes_to_vector(
      nn::models::SsdDefaultBoxes(cells, std::vector<double>(sizes.begin(), sizes.end()),
                                  std::vector<double>(steps.begin(), steps.end()), per_cell));
}
using nn::models::SSD;
using nn::models::SsdNormalize;
using nn::models::SsdOutputs;
struct SsdLossOutputs {
  nn::Tensor loss;
  std::vector<char> selected_negatives;
};
inline SsdLossOutputs SsdLoss(const nn::Tensor &logits, const nn::Tensor &deltas,
                              const std::vector<int> &labels,
                              const std::vector<float> &targets, int classes) {
  if (logits.dim() != 3 || logits.shape()[2] != classes)
    throw std::invalid_argument("SSD class count mismatch");
  const auto N = logits.shape()[0], A = logits.shape()[1];
  const auto result = nn::models::SsdMultiboxLoss(
      logits, deltas,
      nn::from_vector(std::vector<std::int64_t>(labels.begin(), labels.end()), {N, A}),
      nn::from_vector(targets, {N, A, 4}).to(logits.dtype()));
  const auto selected = result.selected_negatives.to_vector<bool>();
  SsdLossOutputs out;
  out.loss = result.loss;
  for (bool chosen : selected)
    out.selected_negatives.push_back(chosen);
  return out;
}
struct SsdTargets {
  std::vector<int> labels;
  std::vector<float> targets;
};
inline SsdTargets MakeSsdTargets(const std::vector<Box> &anchors,
                                 const DetectionScenes &scenes, const std::vector<int> &rows,
                                 int background) {
  if (rows.empty())
    return {};
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  std::vector<std::int64_t> selected;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto row = rows[i];
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("SSD scene row outside source");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
    selected.push_back(std::int64_t(i));
  }
  const auto targets =
      nn::models::MakeSsdTargets(nn::boxes_from_vector(anchors), data, selected, background);
  return {targets.labels.to_vector<int>(), targets.targets.to_vector<float>()};
}

/// Top 200 default boxes above .05 per image and foreground class, decoded and clipped; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection>
DetectSsd(const Forward &forward, const std::vector<Box> &anchors,
          const DetectionScenes &scenes, const std::vector<int> &rows, int classes) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> out;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto net = forward(detector::SceneImages(scenes, chunk));
    if (net.probabilities.shape()[2] != classes)
      throw std::invalid_argument("SSD class count mismatch");
    const auto detected = nn::models::DecodeSsdDetections(
        net.probabilities, net.deltas, nn::boxes_from_vector(anchors).to(net.logits.dtype()),
        nn::from_vector(std::vector<std::int64_t>(chunk.begin(), chunk.end())), scenes.height,
        scenes.width);
    for (const auto &d : nn::models::DetectionEvaluationRecords(detected))
      out.push_back({d.image, d.label, d.score, d.box});
  }
  return out;
}
} // namespace ssd
