#pragma once
/// @file openpose_models.hpp
/// @brief Complete six-stage parallel confidence/PAF OpenPose with explicit raw field outputs.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own every sequential confidence/limb head and all retained OpenPose activations.
///
/// Raw fields are linear unbounded activations,not probabilities. Handles
/// retain full graphs;both preceding branches feed every later refinement.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(197);
/// const auto out = m::OpenPose(p, n::zeros({1, 16, 24, 3}), 8);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO deployment graph, https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt. Source teaching variant uses He initialization from scratch and divisor8 compact widths,not imported pretrained weights or paper-level accuracy/realtime performance.
/// @see OpenPose()
struct OpenPoseResult {
  std::array<nn::Tensor, 6>
      heatmaps; ///< Six raw confidence tensors[N,H/8,W/8,19],including background.
  std::array<nn::Tensor, 6> pafs; ///< Six raw directed-limb tensors[N,H/8,W/8,38].
  std::vector<nn::Tensor>
      stages; ///< All100 backbone,pool,concatenation and branch activations.
};

/// @brief Predict six parallel confidence and directed-limb refinement stages from RGB alone.
///
/// Fixed published input BGR*(255/256)-.5. VGG-shaped ten convolutions plus
/// two CPM convolutions use SAME3x3/ReLU,with three VALID2x2 max pools.
/// First-stage branches use3x3/3x3/3x3/1x1/1x1;five later stages use five7x7
/// then two1x1 convolutions. Later input concatenates previous PAF,previous
/// confidence and shared features. Only terminal19/38 channel counts avoid
/// width division;no BN or imported pretrained backbone. All92 convolutions
/// are learned with He-normal kernels/zero biases. Spatial outputs floor H/W
/// through three pools. Caller supplies explicit intermediate objectives and
/// optimizer;GT,person boxes and IDs are never forward inputs.
///
/// @param p Mutable learned store;warm all184 leaves before optimizer construction.
/// @param rgb Floating[N,H,W,3],N>0,H/W>=8,matching p dtype.
/// @param divisor Positive divisor of64 and128,default1;compact8.
/// @return Six19-channel confidence and38-channel PAF heads plus100 stage activations.
/// @throws std::invalid_argument For invalid NHWC geometry,dtype,width divisor or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(197);
/// const auto out = m::OpenPose(p, n::zeros({1, 16, 24, 3}), 8);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO deployment graph, https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt. Source teaching variant uses He initialization from scratch and divisor8 compact widths,not imported pretrained weights or paper-level accuracy/realtime performance.
/// @see OpenPoseResult
inline OpenPoseResult OpenPose(nn::ParameterStore &p, const nn::Tensor &rgb, int divisor = 1) {
  if (rgb.dim() != 4 || rgb.size(0) < 1 || rgb.size(1) < 8 || rgb.size(2) < 8 ||
      rgb.size(3) != 3 || rgb.dtype() != p.dtype() || divisor < 1 || 64 % divisor ||
      128 % divisor)
    throw std::invalid_argument("OpenPose width divisor must divide64 and128");
  /// @brief Create/reuse one He-normal SAME convolution and optional ReLU.
  /// @param name Unique learned layer prefix.
  /// @param input Floating NHWC branch features.
  /// @param out Output channels,not divided for terminal field layers.
  /// @param kernel Square convolution size.
  /// @param activate Apply ReLU,default true;false at terminal field heads.
  /// @return Differentiable biased convolution activation.
  auto conv = [&](const std::string &name, const nn::Tensor &input, std::int64_t out,
                  std::int64_t kernel, bool activate = true) {
    nn::ConvLayerOptions config;
    config.init = nn::init::he_normal();
    config.activation = activate ? nn::Activation::Relu : nn::Activation::None;
    return nn::conv2d(p, name, input, out, kernel, config);
  };
  OpenPoseResult result;
  // Published input convention: BGR bytes/256-.5, starting from RGB[0,1].
  nn::Tensor x = nn::gather(rgb, nn::from_vector(std::vector<std::int64_t>{2, 1, 0}, {3}), 3) *
                     (255. / 256) -
                 .5;
  const int channels[] = {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 256, 128};
  for (int layer = 0; layer < 12; ++layer) {
    x = conv("backbone/" + std::to_string(layer), x, channels[layer] / divisor, 3);
    result.stages.push_back(x);
    if (layer == 1 || layer == 3 || layer == 7) {
      x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
      result.stages.push_back(x);
    }
  }
  const nn::Tensor features = x;
  for (int stage = 0; stage < 6; ++stage) {
    nn::Tensor input = features;
    if (stage > 0) {
      input = nn::concat({result.pafs[std::size_t(stage - 1)],
                          result.heatmaps[std::size_t(stage - 1)], features},
                         3);
      result.stages.push_back(input);
    }
    for (int branch = 0; branch < 2; ++branch) {
      x = input;
      const int layers = stage == 0 ? 5 : 7;
      for (int layer = 0; layer < layers; ++layer) {
        const bool final = layer + 1 == layers;
        const int out =
            final ? (branch == 0 ? 38 : 19) : (stage == 0 && layer == 3 ? 512 : 128) / divisor;
        const int kernel = stage == 0 ? (layer < 3 ? 3 : 1) : (layer < 5 ? 7 : 1);
        x = conv("stage" + std::to_string(stage) + "/" + (branch == 0 ? "paf/" : "heat/") +
                     std::to_string(layer),
                 x, out, kernel, !final);
        result.stages.push_back(x);
      }
      (branch == 0 ? result.pafs : result.heatmaps)[std::size_t(stage)] = x;
    }
  }
  return result;
}

} // namespace datamunge::nn::models
