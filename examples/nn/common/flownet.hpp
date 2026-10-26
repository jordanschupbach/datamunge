// FlowNetS with six encoder scales and learned feature/flow decoding to scale 2:
// a port of tflow's common/nn/flownet.h.
#pragma once

#include <array>
#include <string>

#include "common/flow_ops.hpp"

namespace flow {

/// Concatenated jointly centred frames; 7/5/5/3/3/3 stride-2 encoder with
/// conv3_1..6_1, predict6, then 4x [deconv, upflow, skip concat, predict].
/// Every pyramid level is input pixels / 20; `flow` is the finest level
/// resized (half-pixel bilinear) to the input size, times 20.
inline FlowResult FlowNetS(nn::ParameterStore& p, const nn::Tensor& first, const nn::Tensor& second, int height = 64, int width = 64,
                           int base = 64) {
  if (height % 64 || width % 64 || height < 64 || width < 64 || base < 1)
    throw std::invalid_argument("FlowNet uses positive width and dimensions divisible by64");
  const auto              centered = FlowCenterPair(first, second);
  nn::Tensor              x        = nn::concat({centered.first, centered.second}, 3);
  FlowResult              result;
  const int               widths[] = {base, base * 2, base * 4, base * 8, base * 8, base * 16};
  std::vector<nn::Tensor> encoder;
  for (int level = 1; level <= 6; ++level) {
    const int kernel = level == 1 ? 7 : level <= 3 ? 5 : 3, out = widths[level - 1];
    x = FlowConv(p, "conv" + std::to_string(level), x, out, kernel, 2);
    result.stages.push_back(x);
    if (level >= 3) {
      x = FlowConv(p, "conv" + std::to_string(level) + "_1", x, out);
      result.stages.push_back(x);
    }
    encoder.push_back(x);
  }
  nn::Tensor flow = FlowConv(p, "predict6", x, 2, 3, 1, false);
  result.pyramid[4] = flow;
  result.stages.push_back(flow);
  for (int level = 5; level >= 2; --level) {
    const int  out     = base * (1 << (level - 2));
    const auto decoded = FlowUp(p, "deconv" + std::to_string(level), x, out);
    const auto upflow  = FlowUp(p, "upflow" + std::to_string(level + 1), flow, 2, false, false);
    result.stages.push_back(decoded);
    result.stages.push_back(upflow);
    x = nn::concat({encoder[std::size_t(level - 1)], decoded, upflow}, 3);
    result.stages.push_back(x);
    flow = FlowConv(p, "predict" + std::to_string(level), x, 2, 3, 1, false);
    result.pyramid[std::size_t(level - 2)] = flow;
    result.stages.push_back(flow);
  }
  result.flow = 20.0 * FlowResizeValues(result.pyramid[0], height, width);
  return result;
}

}  // namespace flow
