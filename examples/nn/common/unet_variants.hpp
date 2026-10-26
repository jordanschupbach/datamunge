// UNet++ nested dense skips and additive attention-gated U-Net (port of
// tflow's common/nn/unet_variants.h).
// https://arxiv.org/abs/1807.10165
// https://arxiv.org/abs/1804.03999
// Nested nodes x[i,j] concatenate x[i,0..j-1] with bilinearly resized
// (corner-aligned) x[i+1,j-1], then apply two 3x3/ReLU convolutions. Four
// full-resolution heads provide deep supervision; inference uses the deepest
// head. Base width 32. Attention U-Net uses learned 2x2 upsampling and
// sigmoid(psi(ReLU(Wx*x+Wg*g))) on each skip, base 64. These demonstrations
// omit BN in the convolution pairs/gates; no pretrained or biomedical
// performance is implied.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/unet.hpp"

namespace unet {

struct NestedUnetConfig {
  int classes = 2, input_channels = 3, base_width = 32, levels = 4;
};
struct NestedUnetOutputs {
  nn::Tensor              logits;
  std::vector<nn::Tensor> heads, nodes;
};

inline NestedUnetOutputs NestedUnet(nn::ParameterStore& p, const nn::Tensor& images, const NestedUnetConfig& cfg = {}) {
  if (cfg.classes < 2 || cfg.input_channels < 1 || cfg.base_width < 1 || cfg.levels < 1 || cfg.levels > 5 ||
      images.shape().back() != cfg.input_channels)
    throw std::invalid_argument("invalid nested U-Net configuration");
  std::vector<std::vector<nn::Tensor>> grid(std::size_t(cfg.levels + 1), std::vector<nn::Tensor>(std::size_t(cfg.levels + 1)));
  NestedUnetOutputs                    result;
  auto node = [](int i, int j) { return "x" + std::to_string(i) + "_" + std::to_string(j); };
  for (int i = 0; i <= cfg.levels; ++i) {
    const std::int64_t width = std::int64_t(cfg.base_width) << i;
    grid[std::size_t(i)][0] =
        ConvPair(p, node(i, 0), i ? nn::max_pool2d(grid[std::size_t(i - 1)][0], 2, 2, nn::Padding::Valid) : images, width);
    result.nodes.push_back(grid[std::size_t(i)][0]);
  }
  for (int j = 1; j <= cfg.levels; ++j)
    for (int i = 0; i + j <= cfg.levels; ++i) {
      const std::int64_t      width = std::int64_t(cfg.base_width) << i;
      std::vector<nn::Tensor> inputs(grid[std::size_t(i)].begin(), grid[std::size_t(i)].begin() + j);
      const auto&             reference = grid[std::size_t(i)][0];
      inputs.push_back(nn::resize_bilinear(grid[std::size_t(i + 1)][std::size_t(j - 1)],
                                           {reference.shape()[1], reference.shape()[2]}, true));
      grid[std::size_t(i)][std::size_t(j)] = ConvPair(p, node(i, j), nn::concat(inputs, 3), width);
      result.nodes.push_back(grid[std::size_t(i)][std::size_t(j)]);
    }
  nn::ConvLayerOptions output;  // linear 1x1, Glorot uniform
  for (int j = 1; j <= cfg.levels; ++j)
    result.heads.push_back(nn::conv2d(p, "head" + std::to_string(j), grid[0][std::size_t(j)], cfg.classes, 1, output));
  result.logits = result.heads.back();
  return result;
}

struct AttentionGateOutputs {
  nn::Tensor coefficients, output;
};
/// Additive gate: coefficients = sigmoid(psi(ReLU(Wx skip + Wg gate))), output = skip * coefficients.
/// 1x1 projections "<name>/x", "<name>/g", "<name>/psi" (W, b; Glorot uniform).
inline AttentionGateOutputs AttentionGate(nn::ParameterStore& p, const std::string& name, const nn::Tensor& skip,
                                          const nn::Tensor& gate, std::int64_t hidden) {
  if (hidden < 1) throw std::invalid_argument("positive attention gate widths required");
  nn::ConvLayerOptions projection;
  const auto           x = nn::conv2d(p, name + "/x", skip, hidden, 1, projection);
  const auto           g = nn::conv2d(p, name + "/g", gate, hidden, 1, projection);
  AttentionGateOutputs r;
  r.coefficients = nn::sigmoid(nn::conv2d(p, name + "/psi", nn::relu(x + g), 1, 1, projection));
  r.output       = skip * r.coefficients;
  return r;
}

struct AttentionUnetConfig {
  int classes = 2, input_channels = 3, base_width = 64, levels = 4;
};
struct AttentionUnetOutputs {
  nn::Tensor              logits;
  std::vector<nn::Tensor> coefficients;
};

inline AttentionUnetOutputs AttentionUnet(nn::ParameterStore& p, const nn::Tensor& images,
                                          const AttentionUnetConfig& cfg = {}) {
  if (cfg.classes < 2 || cfg.input_channels < 1 || cfg.base_width < 2 || cfg.levels < 1 || cfg.levels > 5 ||
      images.shape().back() != cfg.input_channels)
    throw std::invalid_argument("invalid attention U-Net configuration");
  std::vector<nn::Tensor> skips;
  nn::Tensor              h = images;
  for (int i = 0; i < cfg.levels; ++i) {
    h = ConvPair(p, "encoder" + std::to_string(i), h, std::int64_t(cfg.base_width) << i);
    skips.push_back(h);
    h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  h = ConvPair(p, "bottleneck", h, 2 * h.shape().back());
  AttentionUnetOutputs result;
  for (int i = cfg.levels - 1; i >= 0; --i) {
    const std::int64_t width = std::int64_t(cfg.base_width) << i;
    const auto         name  = "decoder" + std::to_string(i);
    h                        = UpConv(p, name + "/up", h, width);
    const auto gate          = AttentionGate(p, name + "/gate", CenterCropTo(skips[std::size_t(i)], h), h, width / 2);
    result.coefficients.push_back(gate.coefficients);
    h = ConvPair(p, name, nn::concat({gate.output, h}, 3), width);
  }
  nn::ConvLayerOptions output;
  result.logits = nn::conv2d(p, "output", h, cfg.classes, 1, output);
  return result;
}

}  // namespace unet
