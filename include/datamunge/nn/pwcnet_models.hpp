#pragma once
/// @file pwcnet_models.hpp
/// @brief Shared tensor feature pyramids,warped81-channel costs,dense flow estimates and dilated
/// context.
#include <cmath>

#include "datamunge/nn/convolution_components.hpp"
#include "datamunge/nn/flow_components.hpp"
#include "datamunge/nn/model_plot.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure Gaussian kernel initialization with an explicit PWC fan-in.
///
/// Returns initializer configuration only;random tensor draws happen when
/// ParameterStore first creates each named leaf. Explicit transpose fan-in
/// uses16*Cin,independent of the stored[4,4,Cout,Cin] channel ordering.
///
/// @param fan_in Positive flattened input connectivity,9*Cin for3x3 or16*Cin for transpose.
/// @return Normal initializer metadata with standard deviation sqrt(2/fan_in).
/// @throws std::invalid_argument For nonpositive fan-in.
/// @par Example
/// @code
/// const auto init = dm::nn::models::PwcHeNormal(9 * 16);
/// @endcode
/// @par References
/// - Sun et al.(2018),PWC-Net:CNNs for Optical Flow Using Pyramid,Warping,and Cost Volume,
/// https://arxiv.org/abs/1709.02371;author graph
/// https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py. Compact
/// channels,explicit fan-in,portable dilated context and input-pixel/20 teaching conventions follow
/// the source example.
/// @see PWCNet()
inline nn::Initializer PwcHeNormal(std::int64_t fan_in) {
  if (fan_in < 1)
    throw std::invalid_argument("positive explicit PWC fan-in required");
  return nn::init::normal(std::sqrt(2.0 / double(fan_in)));
}

/// @brief Predict optical flow with shared pyramids,warped cost volumes and dense decoding.
///
/// Six feature levels share the same18 convolutions across both observed images.
/// At levels6..2,radius4 dy-major81-channel costs feed five dense estimators.
/// Finer levels backward-warp second-image features using20/(2^level) converted
/// flow,then concatenate costs,first features,learned upflow and upfeature.
/// Seven context dilations1,2,4,8,16,1,1 refine level2 with a residual vector.
/// Explicit fan-in Gaussian kernels,LeakyReLU.1,zero biases,no BN or centering.
/// Portable zero-inserted dilated kernels preserve all derivatives. Inference
/// accepts two RGB frames only;no target/mask/hidden optimizer. All internal flow
/// values retain input-image pixels/20 until final half-pixel resize times20.
///
/// During architecture tracing, cost volumes and warps become blocks with both
/// feature/flow inputs. Primitive detail still exposes every displacement and sample.
///
/// @param p Mutable learned store,warm all126 leaves before optimizer construction.
/// @param first Floating[N,height,width,3] observed first RGB frame[0,1],positive N.
/// @param second Matching observed second RGB frame,same dtype.
/// @param height Positive multiple of64,input height,default64.
/// @param width Positive multiple of64,input width,default64.
/// @param compact Use reduced widths while retaining all graph stages,default false.
/// @return Full input-pixel flow,five fixed input-pixel/20 pyramid fields and exposed stages.
/// @throws std::invalid_argument For invalid dimensions,dtypes or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(163);
/// const auto out = n::models::PWCNet(p, n::ones({1, 64, 64, 3}) * .5,
///                                    n::ones({1, 64, 64, 3}) * .4, 64, 64, true);
/// @endcode
/// @par References
/// - Sun et al.(2018),PWC-Net:CNNs for Optical Flow Using Pyramid,Warping,and Cost Volume,
/// https://arxiv.org/abs/1709.02371;author graph
/// https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py. Compact
/// channels,explicit fan-in,portable dilated context and input-pixel/20 teaching conventions follow
/// the source example.
/// @see PwcHeNormal(), FlowWarp(), FlowCorrelation(), FlowObjective()
inline FlowResult PWCNet(
    nn::ParameterStore& p,
    const nn::Tensor&   first,
    const nn::Tensor&   second,
    int                 height  = 64,
    int                 width   = 64,
    bool                compact = false) {
  if (height < 64 || width < 64 || height % 64 || width % 64 || first.dim() != 4 ||
      first.size(0) < 1 || first.shape() != nn::Shape{first.size(0), height, width, 3} ||
      second.shape() != first.shape() || first.dtype() != p.dtype() || second.dtype() != p.dtype())
    throw std::invalid_argument("PWC requires dimensions divisible by64");
  const std::array<int, 6> channels = compact ? std::array<int, 6>{4, 8, 12, 16, 24, 32}
                                              : std::array<int, 6>{16, 32, 64, 96, 128, 196};
  const std::array<int, 5> dense =
      compact ? std::array<int, 5>{16, 16, 12, 8, 4} : std::array<int, 5>{128, 128, 96, 64, 32};
  FlowResult                             result;
  std::array<std::vector<nn::Tensor>, 2> features;
  for (int image = 0; image < 2; ++image) {
    nn::Tensor   x  = image ? second : first;
    std::int64_t in = 3;
    for (int level = 1; level <= 6; ++level) {
      for (int layer = 0; layer < 3; ++layer) {
        const auto name = "pyramid/" + std::to_string(level) + "/" + std::to_string(layer);
        x               = FlowConv(
            p,
            name,
            x,
            channels[std::size_t(level - 1)],
            3,
            layer == 0 ? 2 : 1,
            true,
            PwcHeNormal(9 * in));
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
      const auto feature = paired, pixels = FlowPyramidPixels(upflow, level);
      paired = FlowWarp(feature, pixels).value;
      if (nn::model_plot_detail::recording())
        paired = nn::model_stage(
            "Level " + std::to_string(level) + " warp",
            std::vector<nn::Tensor>{feature, pixels},
            paired);
      result.stages.push_back(paired);
    }
    nn::Tensor cost = nn::leaky_relu(FlowCorrelation(features[0][l], paired, 4), .1);
    if (nn::model_plot_detail::recording())
      cost = nn::model_stage(
          "Level " + std::to_string(level) + " cost volume",
          std::vector<nn::Tensor>{features[0][l], paired},
          cost);
    result.stages.push_back(cost);
    x  = cost;
    in = 81;
    if (level < 6) {
      x = nn::concat({cost, features[0][l], upflow, upfeature}, 3);
      in += channels[l] + 4;
    }
    for (int layer = 0; layer < 5; ++layer) {
      const auto next = FlowConv(
          p,
          "estimate/" + std::to_string(level) + "/" + std::to_string(layer),
          x,
          dense[std::size_t(layer)],
          3,
          1,
          true,
          PwcHeNormal(9 * in));
      x = nn::concat({next, x}, 3);
      in += dense[std::size_t(layer)];
      result.stages.push_back(x);
    }
    flow = FlowConv(p, "predict/" + std::to_string(level), x, 2, 3, 1, false, PwcHeNormal(9 * in));
    result.pyramid[std::size_t(level - 2)] = flow;
    result.stages.push_back(flow);
    if (level > 2) {
      upflow =
          FlowUp(p, "upflow/" + std::to_string(level), flow, 2, false, true, PwcHeNormal(16 * 2));
      upfeature =
          FlowUp(p, "upfeature/" + std::to_string(level), x, 2, false, true, PwcHeNormal(16 * in));
      result.stages.push_back(upflow);
      result.stages.push_back(upfeature);
    }
  }
  /// @brief Context dilation metadata,retaining all seven reference stages.
  const int                rates[] = {1, 2, 4, 8, 16, 1, 1};
  const std::array<int, 7> context = compact ? std::array<int, 7>{16, 16, 16, 12, 8, 4, 2}
                                             : std::array<int, 7>{128, 128, 128, 96, 64, 32, 2};
  for (int layer = 0; layer < 7; ++layer) {
    const int        out     = context[std::size_t(layer)];
    const auto       name    = "context/" + std::to_string(layer);
    const nn::Tensor weights = p.get(name + "/W", {3, 3, in, out}, PwcHeNormal(9 * in));
    x                        = nn::bias_add(
        nn::conv2d(
            x,
            nn::expand_dilated_kernel(weights, rates[layer]),
            nn::conv_options(1, nn::Padding::Same)),
        p.get(name + "/b", {out}, nn::init::zeros()));
    if (layer < 6)
      x = nn::leaky_relu(x, .1);
    in = out;
    result.stages.push_back(x);
  }
  result.pyramid[0] = flow + x;
  result.stages.push_back(result.pyramid[0]);
  result.flow = 20.0 * FlowResizeValues(result.pyramid[0], height, width);
  return result;
}

} // namespace datamunge::nn::models
