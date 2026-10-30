#pragma once
// Compatibility containers forward all model,loss,alignment,target and decode numerics.
#include "common/instance_segmentation.hpp"
#include "common/two_stage.hpp"
#include "datamunge/nn/mask_rcnn_models.hpp"
namespace maskrcnn {
namespace nn = datamunge::nn;
using detector::Box;
using detector::Detection;
inline nn::Tensor AlignCrops(const std::vector<Box> &boxes, int cells, int stride, int bins) {
  return nn::models::MaskAlignmentBoxes(nn::boxes_from_vector(boxes), cells, cells, stride,
                                        bins);
}
inline nn::Tensor RoiAlign(const nn::Tensor &feature, const std::vector<Box> &boxes,
                           const std::vector<int> &indices, int cells, int stride, int bins) {
  if (feature.dim() != 4 || feature.shape()[1] != cells || feature.shape()[2] != cells)
    throw std::invalid_argument("RoIAlign cells mismatch");
  return nn::models::MaskRoiAlign(
      feature, nn::boxes_from_vector(boxes).to(feature.dtype()),
      nn::from_vector(std::vector<std::int64_t>(indices.begin(), indices.end())), stride,
      bins);
}
using nn::models::MaskRcnnHead;
inline nn::Tensor MaskHead(nn::ParameterStore &p, const nn::Tensor &pooled, int width = 256,
                           int classes = 80) {
  return nn::models::MaskRcnnHead(p, pooled, width, classes);
}
inline nn::Tensor SelectedMaskLoss(const nn::Tensor &logits, const std::vector<int> &labels,
                                   const nn::Tensor &targets, int classes) {
  if (logits.dim() != 4 || logits.shape()[3] != classes)
    throw std::invalid_argument("mask class count mismatch");
  return nn::models::SelectedInstanceMaskLoss(
      logits, nn::from_vector(std::vector<std::int64_t>(labels.begin(), labels.end())),
      targets);
}
inline detector::RoiHeadOutputs PooledRoiHead(nn::ParameterStore &p, const nn::Tensor &pooled,
                                              int classes, int hidden = 256) {
  return nn::models::PooledDetectionRoiHead(p, pooled, classes, hidden);
}
inline std::vector<Detection> DecodePixelRois(const detector::RoiHeadOutputs &out,
                                              const std::vector<int> &rows,
                                              const std::vector<std::pair<int, Box>> &boxes,
                                              int classes) {
  if (boxes.empty())
    return {};
  if (out.probabilities.dim() != 2 || out.probabilities.shape()[1] != classes)
    throw std::invalid_argument("RoI decode classes mismatch");
  std::vector<Box> pixels;
  std::vector<std::int64_t> ids;
  for (const auto &[index, box] : boxes) {
    pixels.push_back(box);
    ids.push_back(rows.at(index));
  }
  const auto decoded = nn::models::DecodeDetectionRoiOutputs(
      out, nn::boxes_from_vector(pixels), nn::from_vector(ids), true, .3, .05, 100, false);
  std::vector<Detection> result;
  for (const auto &d : nn::models::DetectionEvaluationRecords(decoded))
    result.push_back({d.image, d.label, d.score, d.box});
  return result;
}
template <typename HeadFn>
std::vector<Detection> DetectPixelRois(const HeadFn &head, const std::vector<int> &rows,
                                       const std::vector<std::pair<int, Box>> &boxes,
                                       int classes) {
  if (boxes.empty())
    return {};
  std::vector<Box> rois;
  std::vector<int> indices;
  for (const auto &[index, box] : boxes) {
    rois.push_back(box);
    indices.push_back(index);
  }
  return DecodePixelRois(head(rois, indices), rows, boxes, classes);
}
struct MaskTargets {
  std::vector<Box> boxes;
  std::vector<int> indices, labels;
  nn::Tensor targets;
};
template <typename Scenes>
MaskTargets PositiveMaskTargets(const Scenes &scenes, const std::vector<int> &rows,
                                const detector::RoiBatch &batch) {
  const auto target = nn::models::PositiveInstanceMaskTargets(
      instance::TensorAnnotations(scenes), std::vector<std::int64_t>(rows.begin(), rows.end()),
      {detector::RoiTensor(batch.rois), detector::RoiTensor(batch.pixels),
       detector::RoiTensor(batch.targets),
       nn::from_vector(std::vector<std::int64_t>(batch.indices.begin(), batch.indices.end())),
       nn::from_vector(std::vector<std::int64_t>(batch.labels.begin(), batch.labels.end())),
       nn::from_vector(batch.mask)});
  return {nn::boxes_to_vector(target.boxes), target.indices.template to_vector<int>(),
          target.labels.template to_vector<int>(), target.targets};
}
} // namespace maskrcnn
