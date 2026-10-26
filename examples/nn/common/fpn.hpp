// Feature pyramid networks (port of tflow's common/nn/fpn.h): lateral 1x1
// projections, top-down nearest upsampling and addition, then a separate 3x3
// smoothing convolution at every level. The unsmoothed inner maps feed the
// next top-down step.
// https://arxiv.org/abs/1612.03144
// https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
//
// Inputs are NHWC features in increasing stride order (C2..C5 in the paper).
// Resize to each lateral map's actual dimensions (TF ResizeNearestNeighbor,
// no corner alignment), including odd sizes. All outputs have the requested
// channel count; no activation or BN is applied. An optional coarser output
// (P6) is a stride-2, 1x1 max pool of the last smoothed feature. Convolutions
// have biases initialized to zero and He-uniform kernels. Parameter names are
// "<name>/level<i+2>/{lateral,smooth}/{W,b}".
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace fpn {

namespace nn = datamunge::nn;

struct FPNOutputs {
  std::vector<nn::Tensor> inner, levels;
};
inline FPNOutputs FeaturePyramid(nn::ParameterStore& p, const std::string& name, const std::vector<nn::Tensor>& inputs,
                                 std::int64_t width = 256, bool extra_level = true) {
  if (inputs.empty() || width < 1) throw std::invalid_argument("FPN requires nonempty inputs and positive width");
  FPNOutputs result;
  result.inner.resize(inputs.size());
  result.levels.resize(inputs.size());
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  for (int i = int(inputs.size()) - 1; i >= 0; --i) {
    const auto prefix = name + "/level" + std::to_string(i + 2);
    const auto k      = std::size_t(i);
    nn::Tensor h      = nn::conv2d(p, prefix + "/lateral", inputs[k], width, 1, cfg);
    if (k + 1 < inputs.size()) h = h + nn::resize_nearest(result.inner[k + 1], {h.shape()[1], h.shape()[2]});
    result.inner[k]  = h;
    result.levels[k] = nn::conv2d(p, prefix + "/smooth", h, width, 3, cfg);
  }
  if (extra_level) result.levels.push_back(nn::max_pool2d(result.levels.back(), 1, 2, nn::Padding::Valid));
  return result;
}

}  // namespace fpn
