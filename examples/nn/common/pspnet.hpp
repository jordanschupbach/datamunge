// PSPNet: dilated ResNet and four-level pyramid pooling (port of tflow's
// common/nn/pspnet.h).
// https://arxiv.org/abs/1612.01105
// Adaptive average bins use floor starts and ceil ends (overlap on odd grids).
// Slice/mean/concat keep exact derivatives, including repeated pixels.
// The backbone is deeplab::DenseResNet (tflow's ResNet with output_stride 8,
// stride on the 3x3), pyramid/head/auxiliary convolutions are
// normalized_conv::NormalizedConv (no bias, BN with gamma, momentum .9, eps 1e-5).
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/deeplab.hpp"
#include "common/normalized_conv.hpp"

namespace pspnet {

namespace nn = datamunge::nn;

/// [N, bins, bins, C] means over bins [floor(i H / b), ceil((i + 1) H / b)) (same for width).
inline nn::Tensor AdaptiveAveragePool(const nn::Tensor& input, int bins) {
  if (bins < 1) throw std::invalid_argument("positive pyramid bin count required");
  if (bins == 1) return nn::mean(input, {1, 2}, true);
  const auto&             s = input.shape();
  const std::int64_t      h = s[1], w = s[2];
  std::vector<nn::Tensor> rows;
  for (int y = 0; y < bins; ++y) {
    std::vector<nn::Tensor> columns;
    for (int x = 0; x < bins; ++x) {
      const std::int64_t y0 = h * y / bins, y1 = (h * (y + 1) + bins - 1) / bins;
      const std::int64_t x0 = w * x / bins, x1 = (w * (x + 1) + bins - 1) / bins;
      columns.push_back(nn::mean(nn::slice(input, {0, y0, x0, 0}, {s[0], y1 - y0, x1 - x0, s[3]}), {1, 2}, true));
    }
    rows.push_back(nn::concat(columns, 2));
  }
  return nn::concat(rows, 1);
}

struct PSPNetConfig {
  int   classes = 21, depth = 101, width_divisor = 1, pool_width = 512, head_width = 512, aux_width = 256;
  float keep_probability = .9f;
  bool  auxiliary        = true;
};
struct PSPNetOutputs {
  nn::Tensor              features, coarse, logits, auxiliary;
  std::vector<nn::Tensor> pools;
};

inline PSPNetOutputs PSPNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images,
                            const PSPNetConfig& cfg = {}, bool training = false) {
  if (cfg.classes < 2 || cfg.pool_width < 1 || cfg.head_width < 1 || cfg.aux_width < 1 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid PSPNet configuration");
  resnet::ResNetConfig backbone;
  backbone.depth              = cfg.depth;
  backbone.width_divisor      = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3      = true;
  const auto    net           = deeplab::DenseResNet(p, state, images, backbone, 8, training);
  PSPNetOutputs r;
  r.features = net.stages[3];
  normalized_conv::ConvNormConfig norm;
  norm.training = training;
  norm.scale    = true;
  norm.momentum = .9;
  norm.epsilon  = 1e-5;
  const nn::ImageSize     feature_size{r.features.shape()[1], r.features.shape()[2]};
  const nn::ImageSize     image_size{images.shape()[1], images.shape()[2]};
  std::vector<nn::Tensor> branches = {r.features};
  for (int bins : {1, 2, 3, 6}) {
    const auto pool = AdaptiveAveragePool(r.features, bins);
    r.pools.push_back(pool);
    const auto project =
        normalized_conv::NormalizedConv(p, state, "pyramid/bin" + std::to_string(bins), pool, cfg.pool_width, 1, 1, norm);
    branches.push_back(nn::resize_bilinear(project, feature_size, true));
  }
  nn::Tensor h = normalized_conv::NormalizedConv(p, state, "head", nn::concat(branches, 3), cfg.head_width, 3, 3, norm);
  if (training && cfg.keep_probability < 1) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  nn::ConvLayerOptions score;  // 1x1, bias, Glorot uniform
  r.coarse = nn::conv2d(p, "classifier", h, cfg.classes, 1, score);
  r.logits = nn::resize_bilinear(r.coarse, image_size, true);
  if (cfg.auxiliary) {
    nn::Tensor aux = normalized_conv::NormalizedConv(p, state, "aux/conv", net.stages[2], cfg.aux_width, 3, 3, norm);
    if (training && cfg.keep_probability < 1) aux = nn::dropout(aux, 1 - cfg.keep_probability, true, p.generator());
    aux         = nn::conv2d(p, "aux/classifier", aux, cfg.classes, 1, score);
    r.auxiliary = nn::resize_bilinear(aux, image_size, true);
  }
  return r;
}

}  // namespace pspnet
