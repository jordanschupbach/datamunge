// DeepLab v3/v3+: five-branch ASPP and low-level separable decoder (port of
// tflow's common/nn/deeplab_v3.h).
// https://arxiv.org/abs/1706.05587
// https://arxiv.org/abs/1802.02611
// Both use a ResNet encoder (supported backbone variant), output stride 16.
// V3 ASPP: 1x1, 3x3 rates 6/12/18, image-global average projection;
// concatenate, 1x1 project and dropout(.1). V3+ uses separable atrous
// branches and projects C2 to 48 channels, fuses at stride 4, and applies two
// separable 3x3 blocks. BN/ReLU follow each depthwise and pointwise
// operation; class scores are linear. Atrous convolutions are native
// nn::conv2d / nn::depthwise_conv2d dilations (tflow expands the filters).
#pragma once

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/deeplab.hpp"
#include "common/normalized_conv.hpp"

namespace deeplab {

using normalized_conv::ConvNormConfig;
using normalized_conv::NormalizedConv;

/// Depthwise 3x3 (rate) -> BN -> ReLU -> pointwise NormalizedConv (BN, ReLU).
inline nn::Tensor AtrousSeparableBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                       const nn::Tensor& input, std::int64_t out, int rate, const ConvNormConfig& cfg) {
  const std::int64_t in = input.shape().back();
  // tflow passes fans 9/9 (one depthwise channel), He uniform: U(+-sqrt(6/9)).
  const nn::Tensor kernel = p.get(name + "/depthwise/W", {3, 3, in, 1}, nn::init::uniform(std::sqrt(6.0 / 9.0)));
  const nn::Tensor depthwise =
      nn::depthwise_conv2d(input, kernel, nn::conv_options(1, nn::Padding::Same, rate));
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = cfg.epsilon;
  bn.scale        = true;
  bn.update_stats = cfg.update_stats;
  const nn::Tensor normalized = nn::batch_norm(p, state, name + "/depthwise/bn", depthwise, cfg.training, bn);
  auto pointwise = cfg;
  pointwise.dilation = 1;
  return NormalizedConv(p, state, name + "/pointwise", nn::relu(normalized), out, 1, 1, pointwise);
}

/// Global mean -> 1x1 NormalizedConv -> corner-aligned bilinear broadcast to the input size.
inline nn::Tensor ImagePoolBranch(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                  const nn::Tensor& input, std::int64_t out, const ConvNormConfig& cfg) {
  auto one = cfg;
  one.dilation = 1;
  const auto pooled    = nn::mean(input, {1, 2}, true);
  const auto projected = NormalizedConv(p, state, name + "/project", pooled, out, 1, 1, one);
  return ResizeLike(projected, input);
}

struct DeepLabV3Config {
  int                classes = 21, backbone_depth = 50, width_divisor = 1, head_width = 256, decoder_width = 256,
     low_width = 48, output_stride = 16;
  bool               decoder = false;
  std::array<int, 3> rates   = {6, 12, 18};
  float              keep_probability = .9f;
};
struct DeepLabV3Outputs {
  nn::Tensor              features, low, coarse, logits;
  std::vector<nn::Tensor> branches;
};

inline DeepLabV3Outputs DeepLabV3(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images,
                                  const DeepLabV3Config& cfg = {}, bool training = false) {
  if (cfg.classes < 2 || cfg.head_width < 1 || cfg.decoder_width < 1 || cfg.low_width < 1 ||
      (cfg.output_stride != 8 && cfg.output_stride != 16) || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid DeepLab v3 configuration");
  resnet::ResNetConfig backbone;
  backbone.depth              = cfg.backbone_depth;
  backbone.width_divisor      = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3      = true;
  const auto       net        = DenseResNet(p, state, images, backbone, cfg.output_stride, training);
  DeepLabV3Outputs result;
  result.features = net.stages[3];
  ConvNormConfig norm;
  norm.training = training;
  norm.scale    = true;
  norm.momentum = .9;
  norm.epsilon  = 1e-5;
  result.branches.push_back(NormalizedConv(p, state, "aspp/pointwise", result.features, cfg.head_width, 1, 1, norm));
  for (int base_rate : cfg.rates) {
    const int rate = base_rate * (16 / cfg.output_stride);
    if (rate < 1) throw std::invalid_argument("ASPP rate must be positive");
    const auto name = "aspp/rate" + std::to_string(base_rate);
    if (cfg.decoder)
      result.branches.push_back(AtrousSeparableBlock(p, state, name, result.features, cfg.head_width, rate, norm));
    else {
      auto atrous     = norm;
      atrous.dilation = rate;
      result.branches.push_back(NormalizedConv(p, state, name, result.features, cfg.head_width, 3, 3, atrous));
    }
  }
  result.branches.push_back(ImagePoolBranch(p, state, "aspp/image_pool", result.features, cfg.head_width, norm));
  nn::Tensor h = NormalizedConv(p, state, "aspp/project", nn::concat(result.branches, 3), cfg.head_width, 1, 1, norm);
  if (training && cfg.keep_probability < 1) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  if (cfg.decoder) {
    result.low = NormalizedConv(p, state, "decoder/low", net.stages[0], cfg.low_width, 1, 1, norm);
    const auto resized = ResizeLike(h, result.low);
    h = AtrousSeparableBlock(p, state, "decoder/conv1", nn::concat({resized, result.low}, 3), cfg.decoder_width, 1, norm);
    h = AtrousSeparableBlock(p, state, "decoder/conv2", h, cfg.decoder_width, 1, norm);
  }
  result.coarse = nn::conv2d(p, "classifier", h, cfg.classes, 1);
  result.logits = ResizeLike(result.coarse, images);
  return result;
}

}  // namespace deeplab
