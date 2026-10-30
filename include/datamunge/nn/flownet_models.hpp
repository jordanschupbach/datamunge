#pragma once
/// @file flownet_models.hpp
/// @brief Six-scale tensor FlowNetS with learned feature/flow transpose decoding.
#include "datamunge/nn/flow_components.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Predict optical flow with the full six-scale FlowNetS tensor graph.
///
/// Jointly centered concatenated images enter7/5/5/3/3/3 stride2 encoders,
/// extra conv3_1..6_1,then four learned feature/flow transpose-and-skip decoders.
/// LeakyReLU.1,Glorot kernels,zero biases,no BN. Internal values ALWAYS use
/// input pixels/20;final resize keeps values before multiplying by20. No target
/// inputs or hidden optimizer. This is the S architecture,not correlation FlowNetC.
///
/// @param p Mutable learned store,warm every leaf before optimizer construction.
/// @param first Floating[N,height,width,3] observed first frame[0,1],positive N.
/// @param second Matching observed second RGB frame.
/// @param height Positive multiple of64,input height,default64.
/// @param width Positive multiple of64,input width,default64.
/// @param base Positive bounded encoder base width,default64;compact8.
/// @return Full-resolution dx/dy input-pixel flow,pyramid levels2..6 and every internal stage.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(163);
/// const auto y = m::FlowNetS(p, n::zeros({2, 64, 64, 3}), n::zeros({2, 64, 64, 3}), 64, 64, 8);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline FlowResult FlowNetS(nn::ParameterStore &p, const nn::Tensor &first,
                           const nn::Tensor &second, int height = 64, int width = 64,
                           int base = 64) {
  if (height % 64 || width % 64 || height < 64 || width < 64 || base < 1 ||
      base > std::numeric_limits<int>::max() / 16 || first.dim() != 4 || first.size(0) < 1 ||
      first.shape() != nn::Shape{first.size(0), height, width, 3} ||
      first.shape() != second.shape() || first.dtype() != p.dtype() ||
      second.dtype() != p.dtype())
    throw std::invalid_argument("FlowNet uses positive width and dimensions divisible by64");
  const auto centered = FlowCenterPair(first, second);
  nn::Tensor x = nn::concat({centered.first, centered.second}, 3);
  FlowResult result;
  /// @brief Six encoder widths,integer architecture metadata only.
  const int widths[] = {base, base * 2, base * 4, base * 8, base * 8, base * 16};
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
    const int out = base * (1 << (level - 2));
    const auto decoded = FlowUp(p, "deconv" + std::to_string(level), x, out);
    const auto upflow = FlowUp(p, "upflow" + std::to_string(level + 1), flow, 2, false, false);
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

} // namespace datamunge::nn::models
