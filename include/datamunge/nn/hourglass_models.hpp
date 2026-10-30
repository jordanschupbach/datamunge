#pragma once
/// @file hourglass_models.hpp
/// @brief Recursive stacked-hourglass pose estimation with explicit BN state and intermediate heads.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Describe square stacked-hourglass architecture metadata.
///
/// Full256px,width256,stem64,two stacks/depth4/17 outputs.
/// Compact128px,width32,stem8 retains both recursive stacks and feedback.
///
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::CompactHourglassConfig();
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
struct HourglassConfig {
  int size = 256;  ///< Square observed RGB side,divisible by4*2^depth.
  int width = 256; ///< Even recursive feature width.
  int stem = 64;   ///< Positive even stem feature width.
  int stacks = 2;  ///< Number of supervised stacks,at least2.
  int depth = 4;   ///< Recursive down/up levels,1..10.
  int joints = 17; ///< Positive linear output heatmap channels.
};
/// @brief Own raw intermediate-supervision heads and every retained graph stage.
///
/// Heads are linear activations,not probabilities. Stage handles retain
/// input/parameter graphs;caller owns optimizer and BN mutation policy.
///
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(173), state(179);
/// const auto out =
///     m::StackedHourglass(p, state, n::zeros({1, 128, 128, 3}), m::CompactHourglassConfig());
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
struct HourglassResult {
  std::vector<nn::Tensor>
      heatmaps; ///< One raw unbounded[N,size/4,size/4,joints] head per stack.
  std::vector<nn::Tensor>
      stages; ///< Stem,recursive outputs,stack features,heads and feedback activations.
};

/// @brief Apply odd He-normal convolution after explicit symmetric zero padding.
///
/// Explicit symmetric phase reproduces the source Torch padding. No
/// activation,BN or optimizer is hidden;creates/reuses named learned leaves.
///
/// @param p Mutable learned store.
/// @param name Unique layer prefix.
/// @param input Floating NHWC features matching p dtype.
/// @param out Positive output channels.
/// @param kernel Positive odd square width,default1.
/// @param stride Positive spatial stride,default1.
/// @return Differentiable biased VALID convolution after(kernel-1)/2 padding.
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(173);
/// const auto y = m::PoseConv(p, "c", n::ones({1, 7, 9, 2}), 4, 3, 2);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
inline nn::Tensor PoseConv(nn::ParameterStore &p, const std::string &name,
                           const nn::Tensor &input, std::int64_t out, std::int64_t kernel = 1,
                           std::int64_t stride = 1) {
  if (out < 1 || kernel < 1 || kernel % 2 == 0 || stride < 1)
    throw std::invalid_argument(
        "positive output/stride and odd pose convolution kernel required");
  const std::int64_t pad = (kernel - 1) / 2;
  nn::ConvLayerOptions config;
  config.stride = stride;
  config.padding = nn::Padding::Valid;
  config.init = nn::init::he_normal();
  const nn::Tensor padded =
      pad ? nn::pad(input, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}) : input;
  return nn::conv2d(p, name, padded, out, kernel, config);
}

/// @brief Apply BN with explicit learned/moving stores and update policy.
///
/// Momentum.9,epsilon1e-5. Warm affine/state leaves before optimizer creation.
/// Inference and validation must use training=false to preserve moving values.
///
/// @param p Mutable gamma/beta store.
/// @param state Mutable separate moving mean/variance store.
/// @param name Unique normalization prefix.
/// @param input Floating NHWC activations.
/// @param training True uses batch moments,false uses moving state.
/// @param update Moving state changes iff training&&update,default true.
/// @return Differentiable BN activations.
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(173), state(179);
/// const auto y = m::PoseBN(p, state, "bn", n::ones({2, 3, 4, 2}), true, false);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
inline nn::Tensor PoseBN(nn::ParameterStore &p, nn::ParameterStore &state,
                         const std::string &name, const nn::Tensor &input, bool training,
                         bool update = true) {
  nn::BatchNormOptions o;
  o.momentum = .9;
  o.eps = 1e-5;
  o.update_stats = update;
  return nn::batch_norm(p, state, name, input, training, o);
}

/// @brief Apply preactivation1x1/3x3/1x1 bottleneck plus identity/projected skip.
///
/// Each convolution sees BN/ReLU preactivation. A1x1 projection appears
/// when input/output widths differ;no activation after residual addition.
///
/// @param p Mutable learned store.
/// @param state Separate mutable BN statistics.
/// @param name Unique residual prefix.
/// @param input Floating NHWC activations.
/// @param out Positive even output width,bottleneck out/2.
/// @param training Use batch moments if true.
/// @param update Update moving statistics iff training,default true.
/// @return Same-spatial residual output,out channels.
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(173), state(179);
/// const auto y = m::PoseResidual(p, state, "r", n::ones({2, 4, 6, 4}), 4, true, false);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
inline nn::Tensor PoseResidual(nn::ParameterStore &p, nn::ParameterStore &state,
                               const std::string &name, const nn::Tensor &input,
                               std::int64_t out, bool training, bool update = true) {
  if (out < 2 || out % 2)
    throw std::invalid_argument("even residual width required");
  const std::int64_t in = input.size(3);
  nn::Tensor x = nn::relu(PoseBN(p, state, name + "/bn1", input, training, update));
  nn::Tensor y = PoseConv(p, name + "/conv1", x, out / 2);
  y = PoseConv(p, name + "/conv2",
               nn::relu(PoseBN(p, state, name + "/bn2", y, training, update)), out / 2, 3);
  y = PoseConv(p, name + "/conv3",
               nn::relu(PoseBN(p, state, name + "/bn3", y, training, update)), out);
  const nn::Tensor skip = in == out ? input : PoseConv(p, name + "/skip", input, out);
  return y + skip;
}

/// @brief Recursively combine same-scale residual and pooled/upsampled lower paths.
///
/// Lower path is VALID2x2 pooling,preactivation residual,recursive module
/// or bottom residual,and another residual. Explicit nearest resizing returns
/// to the supplied upper geometry. Appends deepest-first recursive stages.
///
/// @param p Mutable learned store.
/// @param state Separate mutable BN store.
/// @param name Unique recursive prefix.
/// @param input Floating NHWC square side-by-side features.
/// @param side Positive extent,divisible by2^depth.
/// @param channels Positive even recursive feature width.
/// @param depth Recursion1..10.
/// @param training Use batch moments if true.
/// @param result Mutable collector;append one output per recursion level.
/// @param update Update moving state iff training,default true.
/// @return Upper residual+nearest-upsampled lower branch.
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(173), state(179);
/// m::HourglassResult r;
/// const auto y =
///     m::HourglassModule(p, state, "hg", n::ones({2, 8, 8, 4}), 8, 4, 2, true, r, false);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
inline nn::Tensor HourglassModule(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const std::string &name, const nn::Tensor &input,
                                  std::int64_t side, std::int64_t channels, int depth,
                                  bool training, HourglassResult &result, bool update = true) {
  if (depth < 1 || depth > 10 || side < (1LL << depth) || side % (1LL << depth) ||
      input.dim() != 4 || input.size(1) != side || input.size(2) != side || channels < 2 ||
      channels % 2)
    throw std::invalid_argument("aligned dyadic hourglass recursion geometry required");
  const nn::Tensor upper =
      PoseResidual(p, state, name + "/upper", input, channels, training, update);
  nn::Tensor lower = nn::max_pool2d(input, 2, 2, nn::Padding::Valid);
  lower = PoseResidual(p, state, name + "/lower1", lower, channels, training, update);
  if (depth > 1)
    lower = HourglassModule(p, state, name + "/inner", lower, side / 2, channels, depth - 1,
                            training, result, update);
  else
    lower = PoseResidual(p, state, name + "/bottom", lower, channels, training, update);
  lower = PoseResidual(p, state, name + "/lower3", lower, channels, training, update);
  const nn::Tensor enlarged = nn::resize_nearest(lower, {side, side});
  nn::Tensor output = upper + enlarged;
  result.stages.push_back(output);
  return output;
}

/// @brief Run complete recursive hourglass stacks with learned feature/prediction feedback.
///
/// Stem7x7/stride2,preactivation bottleneck,pool2 and two bottlenecks yield
/// quarter-resolution features. Every stack performs recursive hourglass,
/// output residual,1x1/BN/ReLU and linear joint head. Before the next stack,
/// add incoming features plus learned feature AND prediction projections.
/// Both stack heads need explicit intermediate supervision in caller loops.
/// No targets,optimizer or fitting enters forward. BN state is separate.
///
/// @param p Mutable learned store;warm every leaf before optimizer construction.
/// @param state Separate mutable moving BN statistics.
/// @param image Floating[N,config.size,config.size,3] observed RGB matching p dtype.
/// @param config Architecture metadata,default full256px.
/// @param training Use batch moments if true,default false.
/// @param update Mutate moving BN iff training&&update,default true.
/// @return One raw linear heatmap head per stack and every retained graph stage.
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(173), state(179);
/// const auto out =
///     m::StackedHourglass(p, state, n::zeros({1, 128, 128, 3}), m::CompactHourglassConfig());
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
inline HourglassResult StackedHourglass(nn::ParameterStore &p, nn::ParameterStore &state,
                                        const nn::Tensor &image,
                                        const HourglassConfig &config = {},
                                        bool training = false, bool update = true) {
  if (config.depth < 1 || config.depth > 10 || image.dim() != 4 || image.size(0) < 1 ||
      image.shape() != nn::Shape{image.size(0), config.size, config.size, 3} ||
      image.dtype() != p.dtype() || config.size < 4 * (1 << config.depth) ||
      config.size % (4 * (1 << config.depth)) || config.stacks < 2 || config.width < 2 ||
      config.width % 2 || config.stem < 2 || config.stem % 2 || config.joints < 1)
    throw std::invalid_argument("invalid stacked hourglass configuration");
  HourglassResult result;
  nn::Tensor x = PoseConv(p, "stem/conv", image, config.stem, 7, 2);
  x = nn::relu(PoseBN(p, state, "stem/bn", x, training, update));
  x = PoseResidual(p, state, "stem/res1", x, config.stem * 2, training, update);
  x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
  x = PoseResidual(p, state, "stem/res2", x, config.stem * 2, training, update);
  x = PoseResidual(p, state, "stem/res3", x, config.width, training, update);
  result.stages.push_back(x);
  for (int stack = 0; stack < config.stacks; ++stack) {
    const auto name = "stack" + std::to_string(stack);
    const auto hg = HourglassModule(p, state, name + "/hourglass", x, config.size / 4,
                                    config.width, config.depth, training, result, update);
    nn::Tensor features =
        PoseResidual(p, state, name + "/output_res", hg, config.width, training, update);
    features = PoseConv(p, name + "/lin", features, config.width);
    features = nn::relu(PoseBN(p, state, name + "/lin_bn", features, training, update));
    result.stages.push_back(features);
    const nn::Tensor heatmap = PoseConv(p, name + "/heatmap", features, config.joints);
    result.heatmaps.push_back(heatmap);
    result.stages.push_back(heatmap);
    if (stack + 1 < config.stacks) {
      const nn::Tensor feature_projection =
          PoseConv(p, name + "/merge_feature", features, config.width);
      const nn::Tensor prediction_projection =
          PoseConv(p, name + "/merge_prediction", heatmap, config.width);
      x = x + feature_projection + prediction_projection;
      result.stages.push_back(x);
    }
  }
  return result;
}

/// @brief Return the declared128px/two-stack depth4 teaching configuration.
///
/// Pure metadata;allocates no weights or optimizer.
///
/// @return Size128,width32,stem8,two stacks,depth4,17 joint channels.
/// @throws std::invalid_argument For invalid dyadic geometry,width/kernel/configuration,dtype or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::CompactHourglassConfig();
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;original Torch model, https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua. Source teaching variant uses COCO17,compact128px width32/stem8,source-image-disjoint small oracle person crops and explicit tensor BN state;not MPII benchmark or pretrained pose accuracy.
/// @see StackedHourglass(), HourglassModule(), PoseResidual()
inline HourglassConfig CompactHourglassConfig() {
  HourglassConfig c;
  c.size = 128;
  c.width = 32;
  c.stem = 8;
  return c;
}
} // namespace datamunge::nn::models
