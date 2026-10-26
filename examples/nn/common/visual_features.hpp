// Frozen full VGG-19 activation taps with ORIGINAL max pooling, not style pools:
// a port of tflow's common/nn/visual_features.h (weights: style::FrozenVgg19,
// common/perceptual.hpp).
#pragma once

#include <datamunge/nn/nn.hpp>
#include <string>
#include <vector>

#include "common/perceptual.hpp"

namespace visual {

namespace nn = datamunge::nn;
using nn::Tensor;

struct VisualFeatures {
  Tensor                   centered, first_pre;
  std::vector<Tensor>      activation;  // block1_conv1, block2_conv2, block3_conv4, block4_conv4, block5_conv4
  std::vector<std::string> names;
};

/// All 16 convolutions of VGG-19 (SAME, ReLU) with 2x2 max pools between blocks.
inline VisualFeatures VggVisualFeatures(const style::FrozenVgg19& model, const Tensor& rgb) {
  VisualFeatures out;
  out.centered = style::VggImageNetInput(rgb);
  Tensor x     = out.centered;
  for (int block = 0; block < 5; ++block) {
    const int depth = style::FrozenVgg19::Depths[std::size_t(block)];
    for (int layer = 0; layer < depth; ++layer) {
      const auto   name = "block" + std::to_string(block + 1) + "_conv" + std::to_string(layer + 1);
      const Tensor pre  = nn::bias_add(nn::conv2d(x, model.constants.at(name + "_W")), model.constants.at(name + "_b"));
      x                 = nn::relu(pre);
      if (block == 0 && layer == 0) {
        out.first_pre = pre;
        out.names.push_back(name);
        out.activation.push_back(x);
      } else if (block > 0 && layer == depth - 1) {
        out.names.push_back(name);
        out.activation.push_back(x);
      }
    }
    if (block < 4) x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
  }
  return out;
}

}  // namespace visual
