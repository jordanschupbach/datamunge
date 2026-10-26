// Panoptic FPN backbone and semantic branch; shared by learned instance heads
// (port of tflow's common/nn/panoptic_fpn.h).
// https://arxiv.org/abs/1901.02446
// ResNet50/FPN P2-P6, GN32 semantic widths 128. P2/P3/P4/P5 have 1/1/2/3
// conv/GN/ReLU stages, with bilinear upsampling (TF ResizeBilinear defaults)
// to stride 4 and summed paths. PanopticRoiAlign routes each pixel box to
// pyramid level clamp(floor(4 + log2(sqrt(wh) / canonical)), 2, 5) and
// RoI-aligns it there. tflow aligns every box on all four levels and masks
// the unselected ones; this port aligns each box only on its own level and
// restores the box order with a gather (same values and gradients).
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/fpn.hpp"
#include "common/mask_rcnn.hpp"
#include "common/resnet.hpp"

namespace panoptic {

namespace nn = datamunge::nn;
using detector::Box;

struct PanopticConfig {
  int  size = 256, depth = 50, divisor = 1, width = 256, semantic_width = 128, groups = 32, semantic_classes = 54;
  bool grayscale = false;
};
struct PanopticFeatures {
  std::vector<nn::Tensor> pyramid;
  nn::Tensor              semantic;
};

inline PanopticFeatures PanopticBackbone(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                         const PanopticConfig& cfg = {}, bool training = false, bool update_stats = true) {
  if (cfg.size < 64 || cfg.size % 32 || cfg.width < 1 || cfg.semantic_width < 1 || cfg.groups < 1 ||
      cfg.semantic_width % cfg.groups || cfg.semantic_classes < 2)
    throw std::invalid_argument("invalid Panoptic FPN configuration");
  resnet::ResNetConfig rc;
  rc.depth              = cfg.depth;
  rc.width_divisor      = cfg.divisor;
  rc.include_classifier = false;
  rc.stride_on_3x3      = true;
  rc.update_stats       = update_stats;
  const auto backbone   = resnet::ResNet(p, state, cfg.grayscale ? nn::tile(image, {1, 1, 1, 3}) : image, rc, training);
  PanopticFeatures out;
  out.pyramid = fpn::FeaturePyramid(p, "fpn", {backbone.stages[0], backbone.stages[1], backbone.stages[2], backbone.stages[3]},
                                    cfg.width, true)
                    .levels;
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  conv.bias = false;
  nn::Tensor sum;
  for (int level = 0; level < 4; ++level) {
    nn::Tensor h = out.pyramid[std::size_t(level)];
    for (int stage = 0; stage < std::max(1, level); ++stage) {
      const auto name = "semantic/level" + std::to_string(level) + "/stage" + std::to_string(stage);
      h               = nn::relu(nn::group_norm(p, name + "/gn", nn::conv2d(p, name, h, cfg.semantic_width, 3, conv), cfg.groups));
      if (level > 0) {
        const auto& target = out.pyramid[std::size_t(level - stage - 1)];
        h                  = nn::resize_bilinear(h, {target.shape()[1], target.shape()[2]});
      }
    }
    sum = level == 0 ? h : sum + h;
  }
  nn::ConvLayerOptions score;
  score.init   = nn::init::he_uniform();
  out.semantic = nn::conv2d(p, "semantic/score", sum, cfg.semantic_classes, 1, score);
  out.semantic = nn::resize_bilinear(out.semantic, {image.shape()[1], image.shape()[2]});
  return out;
}

/// Pyramid level (2..5) of each pixel box, in float32 as TF computes it.
inline std::vector<int> PanopticLevels(const std::vector<Box>& boxes, float canonical) {
  std::vector<int> levels;
  for (const auto& b : boxes) {
    const float side  = std::sqrt(std::max((b[2] - b[0]) * (b[3] - b[1]), 1e-6f));
    const float level = std::floor(4.f + std::log(side / canonical) / float(std::log(2.)));
    levels.push_back(int(std::clamp(level, 2.f, 5.f)));
  }
  return levels;
}

/// Multi-level RoIAlign of pixel boxes on P2..P5 (`features` [N, size/4.., ...]) -> [K, bins, bins, C].
inline nn::Tensor PanopticRoiAlign(const std::vector<nn::Tensor>& features, const std::vector<Box>& boxes,
                                   const std::vector<int>& indices, int image_size, int bins, float canonical = 224) {
  if (features.size() < 4 || image_size < 64 || !(canonical > 0)) throw std::invalid_argument("four aligned pyramid levels required");
  const auto              levels = PanopticLevels(boxes, canonical);
  std::vector<nn::Tensor> parts;
  std::vector<int>        order;
  for (int i = 0; i < 4; ++i) {
    std::vector<Box> selected;
    std::vector<int> selected_indices;
    for (std::size_t k = 0; k < boxes.size(); ++k)
      if (levels[k] == i + 2) {
        selected.push_back(boxes[k]);
        selected_indices.push_back(indices[k]);
        order.push_back(int(k));
      }
    if (selected.empty()) continue;
    parts.push_back(maskrcnn::RoiAlign(features[std::size_t(i)], selected, selected_indices, image_size / (4 << i), 4 << i, bins));
  }
  if (parts.empty()) return nn::zeros({0, bins, bins, features[0].shape()[3]});
  const nn::Tensor grouped = parts.size() == 1 ? parts[0] : nn::concat(parts, 0);
  std::vector<std::int32_t> inverse(order.size());
  for (std::size_t j = 0; j < order.size(); ++j) inverse[std::size_t(order[j])] = std::int32_t(j);
  return nn::gather(grouped, nn::from_vector(inverse), 0);
}

}  // namespace panoptic
