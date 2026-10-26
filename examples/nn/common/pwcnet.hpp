// PWC-DC: shared feature pyramids, warped cost volumes, dense flow and context:
// a port of tflow's common/nn/pwcnet.h.
// TF -> datamunge: He-normal kernels use tflow's explicit fan-in (k*k*in; for
// the 4x4 transposes [4,4,out,in] that is 16*in, not nn::init::he_normal's
// shape[2]); the dilated context uses dilated::PortableDilatedConv2D like TF.
#pragma once

#include <array>
#include <cmath>
#include <string>

#include "common/dilated.hpp"
#include "common/flow_ops.hpp"

namespace flow {

/// N(0, 2 / fan_in) with tflow's explicit fan-in.
inline nn::Initializer PwcHeNormal(std::int64_t fan_in) { return nn::init::normal(std::sqrt(2.0 / double(fan_in))); }

/// Six-level shared pyramid (3 convs/level, first stride 2), coarse-to-fine
/// radius-4 cost volumes on the backward-warped second pyramid, DenseNet flow
/// estimators, learned upflow/upfeature, and a dilated context network.
/// Every pyramid flow is input pixels / 20; `flow` is the level-2 field
/// resized (half-pixel bilinear) to the input size, times 20.
inline FlowResult PWCNet(nn::ParameterStore& p, const nn::Tensor& first, const nn::Tensor& second, int height = 64, int width = 64,
                         bool compact = false) {
  if (height < 64 || width < 64 || height % 64 || width % 64) throw std::invalid_argument("PWC requires dimensions divisible by64");
  const std::array<int, 6> channels = compact ? std::array<int, 6>{4, 8, 12, 16, 24, 32} : std::array<int, 6>{16, 32, 64, 96, 128, 196};
  const std::array<int, 5> dense    = compact ? std::array<int, 5>{16, 16, 12, 8, 4} : std::array<int, 5>{128, 128, 96, 64, 32};
  FlowResult                             result;
  std::array<std::vector<nn::Tensor>, 2> features;
  for (int image = 0; image < 2; ++image) {
    nn::Tensor x  = image ? second : first;
    std::int64_t in = 3;
    for (int level = 1; level <= 6; ++level) {
      for (int layer = 0; layer < 3; ++layer) {
        const auto name = "pyramid/" + std::to_string(level) + "/" + std::to_string(layer);
        x  = FlowConv(p, name, x, channels[std::size_t(level - 1)], 3, layer == 0 ? 2 : 1, true, PwcHeNormal(9 * in));
        in = channels[std::size_t(level - 1)];
      }
      features[std::size_t(image)].push_back(x);
      result.stages.push_back(x);
    }
  }
  nn::Tensor   flow, upflow, upfeature, x;
  std::int64_t in = 0;
  for (int level = 6; level >= 2; --level) {
    const std::size_t l      = std::size_t(level - 1);
    nn::Tensor        paired = features[1][l];
    if (level < 6) {
      paired = FlowWarp(paired, FlowPyramidPixels(upflow, level)).value;
      result.stages.push_back(paired);
    }
    const nn::Tensor cost = nn::leaky_relu(FlowCorrelation(features[0][l], paired, 4), .1);
    result.stages.push_back(cost);
    x  = cost;
    in = 81;
    if (level < 6) {
      x = nn::concat({cost, features[0][l], upflow, upfeature}, 3);
      in += channels[l] + 4;
    }
    for (int layer = 0; layer < 5; ++layer) {
      const auto next = FlowConv(p, "estimate/" + std::to_string(level) + "/" + std::to_string(layer), x, dense[std::size_t(layer)], 3, 1, true,
                                 PwcHeNormal(9 * in));
      x = nn::concat({next, x}, 3);
      in += dense[std::size_t(layer)];
      result.stages.push_back(x);
    }
    flow                                  = FlowConv(p, "predict/" + std::to_string(level), x, 2, 3, 1, false, PwcHeNormal(9 * in));
    result.pyramid[std::size_t(level - 2)] = flow;
    result.stages.push_back(flow);
    if (level > 2) {
      upflow    = FlowUp(p, "upflow/" + std::to_string(level), flow, 2, false, true, PwcHeNormal(16 * 2));
      upfeature = FlowUp(p, "upfeature/" + std::to_string(level), x, 2, false, true, PwcHeNormal(16 * in));
      result.stages.push_back(upflow);
      result.stages.push_back(upfeature);
    }
  }
  const int                rates[] = {1, 2, 4, 8, 16, 1, 1};
  const std::array<int, 7> context = compact ? std::array<int, 7>{16, 16, 16, 12, 8, 4, 2} : std::array<int, 7>{128, 128, 128, 96, 64, 32, 2};
  for (int layer = 0; layer < 7; ++layer) {
    const int        out     = context[std::size_t(layer)];
    const auto       name    = "context/" + std::to_string(layer);
    const nn::Tensor weights = p.get(name + "/W", {3, 3, in, out}, PwcHeNormal(9 * in));
    x = nn::bias_add(dilated::PortableDilatedConv2D(x, weights, rates[layer]), p.get(name + "/b", {out}, nn::init::zeros()));
    if (layer < 6) x = nn::leaky_relu(x, .1);
    in = out;
    result.stages.push_back(x);
  }
  result.pyramid[0] = flow + x;
  result.stages.push_back(result.pyramid[0]);
  result.flow = 20.0 * FlowResizeValues(result.pyramid[0], height, width);
  return result;
}

}  // namespace flow
