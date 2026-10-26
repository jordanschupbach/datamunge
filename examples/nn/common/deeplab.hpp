// DeepLab-LargeFOV v1 and ResNet/ASPP v2, output stride 8 (port of tflow's
// common/nn/deeplab.h), plus the dense (atrous) ResNet variant of tflow's
// common/nn/resnet.h (output_stride 8/16) that DeepLab v2/v3 use.
// https://arxiv.org/abs/1412.7062
// https://arxiv.org/abs/1606.00915
// V1: VGG-16 layout, first three pools stride 2, remaining pools stride 1,
// conv5 rate 2, 3x3 fc6 rate 12 / 1024 channels, 1x1 fc7 and class score.
// V2: ResNet-101 with later stride suppression / rates 2, 4; four parallel
// score convolutions at rates 6/12/18/24 summed. Bilinear resize (corner
// aligned) restores input dimensions.
//
// TF -> datamunge: nn::conv2d differentiates dilated convolutions natively,
// so atrous layers call it directly (tflow expands the filter with zeros
// because TF's CPU backprop rejects dilation; the results are identical).
// The dense ResNet reproduces tflow's ResNet(output_stride) exactly: once the
// running stride reaches output_stride, later stages keep stride 1 and double
// the dilation, every 3x3 convolution padding symmetrically by its dilation.
// Parameter names and initializers follow tflow (resnet.hpp's ResNet is the
// output_stride 32 case and is reused where no dilation is needed).
#pragma once

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/resnet.hpp"

namespace deeplab {

namespace nn = datamunge::nn;

// ---- dense (atrous) ResNet -------------------------------------------------------------------

/// conv2d (no bias, He uniform, symmetric padding (k/2) * dilation, VALID) -> BN -> optional ReLU.
inline nn::Tensor DenseResNetConv(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                  const nn::Tensor& input, std::int64_t out, std::int64_t kernel,
                                  const resnet::ConvNormConfig& cfg, int dilation, std::int64_t stride = 1,
                                  bool relu = true) {
  const std::int64_t in = input.shape().back();
  const nn::Tensor   w  = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  nn::Conv2DOptions  o  = nn::conv_options(stride, nn::Padding::Valid, kernel > 1 ? dilation : 1);
  if (kernel > 1) {
    const std::int64_t pad = (kernel / 2) * dilation;
    o.padding              = nn::Padding::Explicit;
    o.explicit_paddings    = {pad, pad, pad, pad};
  }
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = cfg.epsilon;
  bn.scale        = cfg.scale;
  bn.update_stats = cfg.update_stats;
  nn::Tensor y = nn::batch_norm(p, state, name + "/bn", nn::conv2d(input, w, o), cfg.training, bn);
  return relu ? nn::relu(y) : y;
}

inline nn::Tensor DenseResNetBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                   const nn::Tensor& input, std::int64_t width, const resnet::ConvNormConfig& cfg,
                                   int dilation, int stride, bool bottleneck, bool stride_on_3x3) {
  const std::int64_t in  = input.shape().back();
  const std::int64_t out = width * (bottleneck ? 4 : 1);
  auto conv = [&](const std::string& part, const nn::Tensor& x, std::int64_t co, std::int64_t k, int step, bool relu) {
    return DenseResNetConv(p, state, name + "/" + part, x, co, k, cfg, dilation, step, relu);
  };
  nn::Tensor h;
  if (bottleneck) {
    h = conv("conv1", input, width, 1, stride_on_3x3 ? 1 : stride, true);
    h = conv("conv2", h, width, 3, stride_on_3x3 ? stride : 1, true);
    h = conv("conv3", h, out, 1, 1, false);
  } else {
    h = conv("conv1", input, width, 3, stride, true);
    h = conv("conv2", h, out, 3, 1, false);
  }
  const nn::Tensor shortcut = in == out && stride == 1 ? input : conv("shortcut", input, out, 1, stride, false);
  return nn::relu(h + shortcut);
}

/// tflow's ResNet with ResNetConfig::output_stride 8/16/32 (no classifier).
inline resnet::ResNetOutputs DenseResNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                         const resnet::ResNetConfig& cfg, int output_stride, bool training) {
  cfg.Validate();
  if (output_stride != 8 && output_stride != 16 && output_stride != 32)
    throw std::invalid_argument("invalid ResNet configuration");
  resnet::ResNetOutputs  out;
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.momentum     = cfg.momentum;
  norm.epsilon      = 1e-5;
  norm.update_stats = cfg.update_stats;
  auto h = DenseResNetConv(p, state, "stem", image, 64 / cfg.width_divisor, cfg.small_stem ? 3 : 7, norm, 1,
                           cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem              = h;
  const bool bottleneck = cfg.depth >= 50;
  const auto counts     = resnet::ResNetBlockCounts(cfg.depth);
  int        current    = cfg.small_stem ? 1 : 4, dilation = 1;
  for (int stage = 0; stage < 4; ++stage) {
    int stage_stride = stage ? 2 : 1;
    if (stage && current >= output_stride) {
      stage_stride = 1;
      dilation *= 2;
    } else
      current *= stage_stride;
    const std::int64_t width = (64 << stage) / cfg.width_divisor;
    for (int block = 0; block < counts[std::size_t(stage)]; ++block) {
      const auto name = "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      h = DenseResNetBlock(p, state, name, h, width, norm, dilation, block == 0 ? stage_stride : 1, bottleneck,
                           cfg.stride_on_3x3);
      out.blocks.push_back(h);
    }
    out.stages[std::size_t(stage)] = h;
  }
  if (cfg.include_classifier) out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

// ---- DeepLab v1 / v2 -------------------------------------------------------------------------

/// Bias + optional ReLU atrous convolution, SAME: "<name>/W" (He uniform), "<name>/b" (zeros).
inline nn::Tensor AtrousLayer(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                              std::int64_t out, std::int64_t kernel, int rate, bool relu = true) {
  const std::int64_t in = input.shape().back();
  const nn::Tensor   w  = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  const nn::Tensor   b  = p.get(name + "/b", {out}, nn::init::zeros());
  const nn::Tensor   y  = nn::bias_add(nn::conv2d(input, w, nn::conv_options(1, nn::Padding::Same, rate)), b);
  return relu ? nn::relu(y) : y;
}

struct DeepLabOutputs {
  nn::Tensor              features, coarse, logits;
  std::vector<nn::Tensor> branches;
};

inline std::vector<nn::Tensor> AtrousScorePyramid(nn::ParameterStore& p, const std::string& name,
                                                  const nn::Tensor& feature, std::int64_t classes,
                                                  const std::array<int, 4>& rates) {
  std::vector<nn::Tensor> outputs;
  for (int rate : rates) {
    if (rate < 1) throw std::invalid_argument("ASPP rates must be positive");
    outputs.push_back(AtrousLayer(p, name + "/rate" + std::to_string(rate), feature, classes, 3, rate, false));
  }
  return outputs;
}

/// Bilinear (corner-aligned) resize of x to the spatial size of `like`.
inline nn::Tensor ResizeLike(const nn::Tensor& x, const nn::Tensor& like) {
  return nn::resize_bilinear(x, {like.shape()[1], like.shape()[2]}, true);
}

struct DeepLabV1Config {
  int   classes = 21, width_divisor = 1, fc_width = 1024, fc_rate = 12;
  float keep_probability = .5f;
};
inline DeepLabOutputs DeepLabV1(nn::ParameterStore& p, const nn::Tensor& image, const DeepLabV1Config& cfg = {},
                                bool training = false) {
  if (cfg.classes < 2 || cfg.width_divisor < 1 || 64 % cfg.width_divisor || cfg.fc_width < 1 || cfg.fc_rate < 1 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid DeepLab v1 configuration");
  DeepLabOutputs r;
  nn::Tensor     h         = image;
  const int      repeats[] = {2, 2, 3, 3, 3};
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t width = (64 << std::min(stage, 3)) / cfg.width_divisor;
    for (int block = 0; block < repeats[stage]; ++block)
      h = AtrousLayer(p, "conv" + std::to_string(stage + 1) + "_" + std::to_string(block + 1), h, width, 3,
                      stage == 4 ? 2 : 1);
    h = nn::max_pool2d(h, 3, stage < 3 ? 2 : 1, nn::Padding::Same);
  }
  r.features = h;
  h          = AtrousLayer(p, "fc6", h, cfg.fc_width, 3, cfg.fc_rate);
  if (training && cfg.keep_probability < 1) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  h = AtrousLayer(p, "fc7", h, cfg.fc_width, 1, 1);
  if (training && cfg.keep_probability < 1) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  r.coarse = AtrousLayer(p, "score", h, cfg.classes, 1, 1, false);
  r.logits = ResizeLike(r.coarse, image);
  return r;
}

struct DeepLabV2Config {
  int                classes = 21, backbone_depth = 101, width_divisor = 1;
  std::array<int, 4> rates = {6, 12, 18, 24};
};
inline DeepLabOutputs DeepLabV2(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                const DeepLabV2Config& cfg = {}, bool training = false) {
  if (cfg.classes < 2) throw std::invalid_argument("DeepLab v2 requires at least two classes");
  resnet::ResNetConfig backbone;
  backbone.depth              = cfg.backbone_depth;
  backbone.width_divisor      = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3      = true;
  const auto     net          = DenseResNet(p, state, image, backbone, 8, training);
  DeepLabOutputs r;
  r.features = net.stages[3];
  r.branches = AtrousScorePyramid(p, "aspp", r.features, cfg.classes, cfg.rates);
  r.coarse   = r.branches[0];
  for (std::size_t i = 1; i < r.branches.size(); ++i) r.coarse = r.coarse + r.branches[i];
  r.logits = ResizeLike(r.coarse, image);
  return r;
}

}  // namespace deeplab
