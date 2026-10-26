// Mask R-CNN: unquantized RoIAlign and class-specific sigmoid mask head (port
// of tflow's common/nn/mask_rcnn.h), plus the pixel-RoI box head of tflow's
// common/ml/two_stage.h (FinishRoiHead with pixel_rois) used by Mask R-CNN
// and Panoptic FPN.
// https://arxiv.org/abs/1703.06870
//
// RoiAlign takes pixel-space boxes and samples 2x2 regular points per bin,
// averaged. The aligned convention subtracts half a feature pixel; symmetric
// 1px mirror padding implements bilinear border replication for boxes near
// the image edge. Boxes are constants (gradients flow to the features only,
// as with TF's CropAndResize).
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/two_stage.hpp"

namespace maskrcnn {

namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;

/// Normalized crop boxes [K, 4] on the 1px-padded feature grid for pixel boxes.
inline nn::Tensor AlignCrops(const std::vector<Box>& boxes, int cells, int stride, int bins) {
  std::vector<float> crops;
  crops.reserve(boxes.size() * 4);
  const float near = .25f / float(bins), far = 1 - .25f / float(bins);
  auto normalize = [&](float start, float extent, float fraction) {
    return (( start + extent * fraction) / float(stride) + .5f) / float(cells + 1);
  };
  for (const auto& b : boxes) {
    const float height = b[2] - b[0], width = b[3] - b[1];
    crops.insert(crops.end(), {normalize(b[0], height, near), normalize(b[1], width, near), normalize(b[0], height, far),
                               normalize(b[1], width, far)});
  }
  return nn::from_vector(crops, {std::int64_t(boxes.size()), 4});
}

/// RoIAlign of `feature` [N, cells, cells, C] (stride `stride`) at pixel boxes -> [K, bins, bins, C].
inline nn::Tensor RoiAlign(const nn::Tensor& feature, const std::vector<Box>& boxes, const std::vector<int>& indices,
                           int cells, int stride, int bins) {
  if (cells < 2 || stride < 1 || bins < 1) throw std::invalid_argument("invalid RoIAlign dimensions");
  const auto padded  = nn::mirror_pad(feature, 1, 1, nn::PadMode::Symmetric);
  const auto sampled = nn::crop_and_resize(padded, AlignCrops(boxes, cells, stride, bins), detector::IndexTensor(indices),
                                           {2 * bins, 2 * bins});
  return nn::avg_pool2d(sampled, 2, 2, nn::Padding::Valid);
}

/// Four 3x3 conv/ReLU, a 2x2 stride-2 transposed conv/ReLU and a linear 1x1 class score.
inline nn::Tensor MaskHead(nn::ParameterStore& p, const nn::Tensor& pooled, int width = 256, int classes = 80) {
  if (width < 1 || classes < 1) throw std::invalid_argument("invalid mask head widths");
  nn::ConvLayerOptions relu;
  relu.activation = nn::Activation::Relu;
  relu.init       = nn::init::he_uniform();
  nn::Tensor h    = pooled;
  for (int i = 0; i < 4; ++i) h = nn::conv2d(p, "mask/conv" + std::to_string(i + 1), h, width, 3, relu);
  auto up    = relu;
  up.stride  = 2;
  up.padding = nn::Padding::Valid;
  h          = nn::conv2d_transpose(p, "mask/up", h, width, 2, up);
  return nn::conv2d(p, "mask/classifier", h, classes, 1);
}

/// Mean sigmoid BCE of the labelled class's mask logits; logits [K, S, S, classes], targets [K, S, S].
inline nn::Tensor SelectedMaskLoss(const nn::Tensor& logits, const std::vector<int>& labels, const nn::Tensor& targets,
                                   int classes) {
  const auto one_hot  = nn::reshape(nn::one_hot(detector::IndexTensor(labels), classes), {-1, 1, 1, classes});
  const auto selected = nn::sum(logits * one_hot, {3});
  return nn::mean(nn::softplus(selected) - selected * targets);
}

/// The dense box head on pooled RoIs [K, 7, 7, C] (tflow FinishRoiHead): fc6/fc7 ReLU, classifier, class deltas.
inline detector::RoiHeadOutputs PooledRoiHead(nn::ParameterStore& p, const nn::Tensor& pooled, int classes,
                                              int hidden = 256) {
  const std::int64_t channels = pooled.shape()[3];
  auto f = nn::dense(p, "head/fc6", nn::reshape(pooled, {-1, 7 * 7 * channels}), hidden, nn::Activation::Relu);
  f      = nn::dense(p, "head/fc7", f, hidden, nn::Activation::Relu);
  detector::RoiHeadOutputs h;
  h.logits        = nn::dense(p, "head/classifier", f, classes);
  h.probabilities = nn::softmax(h.logits, -1);
  h.deltas        = nn::reshape(nn::dense(p, "head/regressor", f, 4 * (classes - 1)), {-1, classes - 1, 4});
  return h;
}

/// Classify and refine pixel boxes [(batch position, box)] with `head(boxes, indices)`; scores above .05.
template <typename HeadFn>
std::vector<Detection> DetectPixelRois(const HeadFn& head, const std::vector<int>& rows,
                                       const std::vector<std::pair<int, Box>>& boxes, int classes) {
  std::vector<Detection> dets;
  if (boxes.empty()) return dets;
  std::vector<Box> rois;
  std::vector<int> indices;
  for (const auto& [index, box] : boxes) {
    rois.push_back(box);
    indices.push_back(index);
  }
  const detector::RoiHeadOutputs out           = head(rois, indices);
  const auto                     probabilities = out.probabilities.template to_vector<float>();
  const auto                     deltas        = out.deltas.template to_vector<float>();
  for (std::size_t k = 0; k < boxes.size(); ++k)
    for (int c = 0; c < classes - 1; ++c) {
      const float score = probabilities[k * std::size_t(classes) + std::size_t(c)];
      if (score <= .05f) continue;
      dets.push_back({rows[std::size_t(boxes[k].first)], c, score,
                      detector::DecodeBox(boxes[k].second, deltas.data() + (k * std::size_t(classes - 1) + std::size_t(c)) * 4)});
    }
  return dets;
}

/// Positive sampled RoIs of a batch with their 28x28 nearest-sampled support targets
/// (each from the ground-truth object it overlaps most).
struct MaskTargets {
  std::vector<Box> boxes;
  std::vector<int> indices, labels;
  nn::Tensor       targets;  // [K, 28, 28]
};
template <typename Scenes>
MaskTargets PositiveMaskTargets(const Scenes& scenes, const std::vector<int>& rows, const detector::RoiBatch& batch) {
  MaskTargets        out;
  std::vector<float> targets;
  for (std::size_t k = 0; k < batch.mask.size(); ++k) {
    if (!batch.mask[k]) continue;
    const int  row = rows[std::size_t(batch.indices[k])];
    const Box& box = batch.pixels[k];
    int        best    = -1;
    float      overlap = -1;
    for (std::size_t j = 0; j < scenes.boxes[std::size_t(row)].size(); ++j) {
      const float iou = detector::BoxIoU(box, scenes.boxes[std::size_t(row)][j]);
      if (iou > overlap) {
        overlap = iou;
        best    = int(j);
      }
    }
    check::Require(best >= 0 && scenes.labels[std::size_t(row)][std::size_t(best)] == batch.labels[k],
                   "positive mask target class agrees with sampled RoI");
    out.boxes.push_back(box);
    out.indices.push_back(batch.indices[k]);
    out.labels.push_back(batch.labels[k]);
    const auto& support = scenes.instance_masks[std::size_t(row)][std::size_t(best)];
    for (int y = 0; y < 28; ++y)
      for (int x = 0; x < 28; ++x) {
        const int yy = std::clamp(int(box[0] + (y + .5f) * (box[2] - box[0]) / 28), 0, scenes.height - 1),
                  xx = std::clamp(int(box[1] + (x + .5f) * (box[3] - box[1]) / 28), 0, scenes.width - 1);
        targets.push_back(float(support[std::size_t(yy * scenes.width + xx)]));
      }
  }
  out.targets = nn::from_vector(targets, {std::int64_t(out.boxes.size()), 28, 28});
  return out;
}

}  // namespace maskrcnn
