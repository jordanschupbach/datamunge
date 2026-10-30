#pragma once
#include "common/fpn.hpp"
#include "common/mask_rcnn.hpp"
#include "common/resnet.hpp"
#include "datamunge/nn/panoptic_fpn_models.hpp"
namespace panoptic {
namespace nn = datamunge::nn;
using detector::Box;
using nn::models::PanopticBackbone;
using nn::models::PanopticConfig;
using nn::models::PanopticFeatures;
inline std::vector<int> PanopticLevels(const std::vector<Box> &boxes, float canonical) {
  return nn::models::PanopticLevels(nn::boxes_from_vector(boxes), canonical).to_vector<int>();
}
inline nn::Tensor PanopticRoiAlign(const std::vector<nn::Tensor> &features,
                                   const std::vector<Box> &boxes,
                                   const std::vector<int> &indices, int image_size, int bins,
                                   float canonical = 224) {
  return nn::models::PanopticRoiAlign(
      features, nn::boxes_from_vector(boxes),
      nn::from_vector(std::vector<std::int64_t>(indices.begin(), indices.end())), image_size,
      bins, canonical);
}
} // namespace panoptic
