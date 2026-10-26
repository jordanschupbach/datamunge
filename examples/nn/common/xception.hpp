// Xception's entry, eight repeated middle blocks, and exit flow (port of
// tflow's common/nn/xception.h).
// https://arxiv.org/abs/1610.02357
// https://github.com/keras-team/keras/blob/master/keras/src/applications/xception.py
//
// Match reference widths, residual projections, padding and activation order.
// A separable operation is depthwise then pointwise, without an intervening
// activation or BN. BN follows the complete operation. The first entry block
// omits its leading ReLU, middle blocks use preactivation, and the final two
// separable layers use postactivation. There is no dropout or auxiliary head.
//
// Initialization uses the Glorot convention for separable filters (TF
// depthwise fans) and He for ordinary convolutions; no pretrained weights are
// imported. BN learns gamma/beta and uses the unbiased EMA variance of
// nn::batch_norm (tflow's FusedBatchNorm convention). Width-scaled
// demonstrations retain all eight middle blocks.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/normalized_conv.hpp"

namespace xception {

namespace nn = datamunge::nn;
using normalized_conv::ConvNormConfig;

struct XceptionConfig {
  int    image_size = 299, classes = 1000, width_divisor = 1;
  double momentum = .99;
  void   Validate() const {
    if (image_size < 71 || classes < 2 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 && width_divisor != 8) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid Xception configuration");
  }
};

struct XceptionOutputs {
  std::vector<std::pair<std::string, nn::Tensor>> stages;
  nn::Tensor                                      logits;
};

/// 3x3 separable convolution (no bias, Glorot) followed by learned-scale BN.
inline nn::Tensor XceptionSeparable(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                    const nn::Tensor& input, std::int64_t out, const ConvNormConfig& cfg) {
  nn::ConvLayerOptions options;
  options.bias = false;
  options.init = nn::init::glorot_uniform();
  const auto           h = nn::separable_conv2d(p, name, input, out, 3, options);
  nn::BatchNormOptions bn;
  bn.momentum     = cfg.momentum;
  bn.eps          = cfg.epsilon;
  bn.scale        = true;
  bn.update_stats = cfg.update_stats;
  return nn::batch_norm(p, state, name + "/bn", h, cfg.training, bn);
}

inline XceptionOutputs Xception(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                const XceptionConfig& cfg = {}, bool training = false) {
  cfg.Validate();
  XceptionOutputs out;
  ConvNormConfig  norm;
  norm.training     = training;
  norm.scale        = true;
  norm.momentum     = cfg.momentum;
  const int divisor = cfg.width_divisor;
  auto convolution  = [&](const std::string& name, const nn::Tensor& x, int channels, int kernel, int stride,
                         const std::string& padding, bool relu) {
    return normalized_conv::NormalizedConv(p, state, name, x, channels, kernel, kernel, norm, stride, padding, relu);
  };
  auto separable = [&](const std::string& name, const nn::Tensor& x, int channels) {
    return XceptionSeparable(p, state, name, x, channels, norm);
  };
  nn::Tensor h = convolution("stem1", image, 32 / divisor, 3, 2, "VALID", true);
  out.stages.emplace_back("stem1", h);
  h = convolution("stem2", h, 64 / divisor, 3, 1, "VALID", true);
  out.stages.emplace_back("stem2", h);
  int channels = 64 / divisor;
  for (int block = 0; block < 3; ++block) {
    const int  width = (block == 0 ? 128 : block == 1 ? 256 : 728) / divisor;
    const auto name  = "entry" + std::to_string(block + 1);
    const auto skip  = convolution(name + "/skip", h, width, 1, 2, "SAME", false);
    if (block > 0) h = nn::relu(h);
    h = separable(name + "/sep1", h, width);
    h = separable(name + "/sep2", nn::relu(h), width);
    h = nn::max_pool2d(h, 3, 2, nn::Padding::Same) + skip;
    channels = width;
    out.stages.emplace_back(name, h);
  }
  for (int block = 0; block < 8; ++block) {
    const auto name = "middle" + std::to_string(block + 1);
    const auto skip = h;
    for (int layer = 1; layer <= 3; ++layer) h = separable(name + "/sep" + std::to_string(layer), nn::relu(h), channels);
    h = h + skip;
    out.stages.emplace_back(name, h);
  }
  const auto skip = convolution("exit/skip", h, 1024 / divisor, 1, 2, "SAME", false);
  h               = separable("exit/sep1", nn::relu(h), channels);
  h               = separable("exit/sep2", nn::relu(h), 1024 / divisor);
  h               = nn::max_pool2d(h, 3, 2, nn::Padding::Same) + skip;
  out.stages.emplace_back("exit", h);
  h = nn::relu(separable("final/sep1", h, 1536 / divisor));
  out.stages.emplace_back("final1", h);
  h = nn::relu(separable("final/sep2", h, 2048 / divisor));
  out.stages.emplace_back("final2", h);
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

}  // namespace xception
