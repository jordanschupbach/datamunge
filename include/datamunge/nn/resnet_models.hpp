#pragma once
/// @file resnet_models.hpp
/// @brief Post-activation basic and bottleneck ResNet tensor models, without fitting loops.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Batch-normalisation mode and settings for explicitly padded ResNet convolutions.
///
/// Value metadata only. The full model enables learned gamma and sets epsilon
/// to 1e-5 and momentum from ResNetConfig; generic component defaults are below.
/// Training moving-state updates can be disabled independently of batch moments.
///
/// @par Example
/// @code
/// dm::nn::models::ResNetNormConfig cfg;
/// cfg.scale = true;
/// cfg.epsilon = 1e-5;
/// @endcode
/// @see ResNetNormalizedConv()
struct ResNetNormConfig {
  bool training = false; ///< Normalize with batch moments when true, moving moments otherwise.
  bool scale = false;    ///< Learn gamma if true, use fixed gamma=1 otherwise.
  bool update_stats = true; ///< Update moving moments only when training and true.
  double momentum = .9997;  ///< Finite old-value EMA retention in [0,1).
  double epsilon = .001;    ///< Finite strictly positive variance stabilizer.
};

/// @brief Apply a bias-free square convolution with symmetric padding, BN and optional ReLU.
///
/// Uses He kernels and no convolution bias. Eager training BN updates population
/// means and unbiased moving variances; inference does not mutate state. Work
/// O(NHoWo kernel² Cin Cout). No fitting; shared mutable stores are not safe
/// for concurrent mutation.
///
/// @param p Mutable floating learnable store; creates/reuses named kernels and BN leaves.
/// @param state Distinct mutable BN moving-moment store, same dtype as p.
/// @param name Prefix for /W and /bn parameter/state names.
/// @param input Nonempty floating NHWC tensor, matching store dtype.
/// @param out Positive output channels.
/// @param kernel Positive square kernel width.
/// @param cfg BN mode, scale, epsilon, momentum and update flag.
/// @param stride Positive common spatial stride.
/// @param pad Nonnegative symmetric spatial zero-padding width; zero means VALID.
/// @param relu Apply final ReLU if true.
/// @return Floating NHWC activation, retaining input/kernel/BN gradient graphs.
/// @throws std::invalid_argument For invalid dimensions, padding, BN settings, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig cfg;
/// cfg.scale = true;
/// cfg.epsilon = 1e-5;
/// const auto y = n::models::ResNetNormalizedConv(p, state, "c", n::ones({2, 5, 5, 3}), 4, 3, cfg,
///                                                2, 1, true);
/// @endcode
/// @par References
/// - He et al. (2016), https://arxiv.org/abs/1512.03385; reference topology, https://github.com/KaimingHe/deep-residual-networks
/// @see ResNetConv()
inline nn::Tensor ResNetNormalizedConv(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const std::string &name, const nn::Tensor &input,
                                       std::int64_t out, std::int64_t kernel,
                                       const ResNetNormConfig &cfg, std::int64_t stride,
                                       std::int64_t pad, bool relu) {
  if (input.dim() != 4 || input.size(0) <= 0 || input.size(1) <= 0 || input.size(2) <= 0 ||
      input.size(3) <= 0 || out <= 0 || kernel <= 0 || stride <= 0 || pad < 0)
    throw std::invalid_argument("invalid normalized ResNet convolution dimensions");
  const std::int64_t in = input.size(3);
  const nn::Tensor w = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Valid);
  if (pad > 0) {
    o.padding = nn::Padding::Explicit;
    o.explicit_paddings = {pad, pad, pad, pad};
  }
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.scale = cfg.scale;
  bn.update_stats = cfg.update_stats;
  nn::Tensor y =
      nn::batch_norm(p, state, name + "/bn", nn::conv2d(input, w, o), cfg.training, bn);
  return relu ? nn::relu(y) : y;
}

/// @brief Return Table 1 residual-block counts for a supported learned depth.
///
/// Depth counts stem, branch convolutions and classifier, excluding shortcut projections.
///
/// @param depth Exactly 18, 34, 50, 101 or 152.
/// @return Four stage counts: 2/2/2/2, 3/4/6/3, 3/4/23/3 or 3/8/36/3.
/// @throws std::invalid_argument For any unsupported depth.
/// @par Example
/// @code
/// const auto counts = dm::nn::models::ResNetBlockCounts(50); // 3,4,6,3
/// @endcode
/// @par References
/// - He et al. (2016), https://arxiv.org/abs/1512.03385; reference topology, https://github.com/KaimingHe/deep-residual-networks
/// @see ResNet()
inline std::array<int, 4> ResNetBlockCounts(int depth) {
  switch (depth) {
  case 18:
    return {2, 2, 2, 2};
  case 34:
  case 50:
    return {3, 4, 6, 3};
  case 101:
    return {3, 4, 23, 3};
  case 152:
    return {3, 8, 36, 3};
  default:
    throw std::invalid_argument("supported ResNet depths are 18/34/50/101/152");
  }
}

/// @brief Apply a ResNet convolution with floor(kernel/2) symmetric padding.
///
/// For odd kernels, stride one preserves spatial size; stride two gives ceil(H/2),
/// ceil(W/2). Explicit symmetric padding fixes edge alignment independently
/// of framework SAME conventions. No optimization occurs here.
///
/// @param p Mutable floating learnable store; creates/reuses named kernels and BN leaves.
/// @param state Distinct mutable BN moving-moment store, same dtype as p.
/// @param name Prefix for /W and /bn parameter/state names.
/// @param input Nonempty floating NHWC tensor, matching store dtype.
/// @param out Positive output channels.
/// @param kernel Positive odd square kernel width, normally one, three or seven.
/// @param cfg BN mode/scale and numerical settings.
/// @param stride Positive common stride, default one.
/// @param relu Apply final ReLU if true (default).
/// @return Floating NHWC activation with retained gradient history.
/// @throws std::invalid_argument For nonpositive/even kernel or invalid layer/tensor/store settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig cfg;
/// cfg.scale = true;
/// cfg.epsilon = 1e-5;
/// const auto y = n::models::ResNetConv(p, state, "c", n::ones({2, 5, 5, 3}), 4, 3, cfg);
/// @endcode
/// @see ResNetNormalizedConv()
inline nn::Tensor ResNetConv(nn::ParameterStore &p, nn::ParameterStore &state,
                             const std::string &name, const nn::Tensor &input,
                             std::int64_t out, std::int64_t kernel,
                             const ResNetNormConfig &cfg, std::int64_t stride = 1,
                             bool relu = true) {
  if (kernel <= 0 || kernel % 2 == 0)
    throw std::invalid_argument("positive odd ResNet kernel required");
  return ResNetNormalizedConv(p, state, name, input, out, kernel, cfg, stride, kernel / 2,
                              relu);
}

/// @brief The two paths, their sum and resulting output of a residual block.
///
/// Handles retain their tensor storage and gradient history. A matching shortcut
/// shares input storage; a dimension-changing shortcut is a learned projection.
/// Post-activation blocks gate the sum with ReLU; preactivation blocks return it directly.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig cfg;
/// cfg.scale = true;
/// cfg.epsilon = 1e-5;
/// const auto out = n::models::ResNetBlock(p, state, "b", n::ones({2, 5, 5, 8}), 2, cfg, 2, true);
/// @endcode
/// @see ResNetBlock()
struct ResNetBlockOutputs {
  nn::Tensor residual;      ///< Learned branch value immediately before shortcut addition.
  nn::Tensor shortcut;      ///< Input identity or learned 1x1 projection.
  nn::Tensor preactivation; ///< Residual plus shortcut, before any output gate.
  nn::Tensor output;        ///< Sum for preactivation units, ReLU(sum) for post-activation units.
};
/// @brief Apply a post-activation basic or fourfold-expanding bottleneck residual block.
///
/// Only channel/stride changes construct a bias-free BN 1x1 shortcut. The last
/// branch BN and projection omit ReLU before addition. A zero residual therefore
/// gives ReLU(shortcut), not an unrestricted identity for signed input. Lazy
/// creation and optional moving-state updates occur, but no fitting loop.
///
/// @param p Mutable floating learnable store; creates/reuses named kernels and BN leaves.
/// @param state Distinct mutable BN moving-moment store, same dtype as p.
/// @param name Prefix for /W and /bn parameter/state names.
/// @param input Nonempty floating NHWC tensor, matching store dtype.
/// @param width Positive branch width; final output channels are width (basic) or 4*width (bottleneck).
/// @param cfg BN mode/scale and numerical settings.
/// @param stride Exactly one (default) or two, for spatial downsampling.
/// @param bottleneck False uses two 3x3 layers; true uses 1x1/3x3/1x1, default false.
/// @param stride_on_3x3 Move bottleneck downsampling to 3x3 if true (v1.5); default false uses first 1x1.
/// @return Residual, shortcut, preactivation and ReLU output maps, retaining their graphs.
/// @throws std::invalid_argument For invalid NHWC dimensions, width, stride, BN settings or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig cfg;
/// cfg.scale = true;
/// cfg.epsilon = 1e-5;
/// const auto out = n::models::ResNetBlock(p, state, "b", n::ones({2, 5, 5, 8}), 2, cfg, 2, true);
/// @endcode
/// @par References
/// - He et al. (2016), https://arxiv.org/abs/1512.03385; reference topology, https://github.com/KaimingHe/deep-residual-networks
/// @see ResNetConfig, ResNet()
inline ResNetBlockOutputs ResNetBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                      const std::string &name, const nn::Tensor &input,
                                      std::int64_t width, const ResNetNormConfig &cfg,
                                      int stride = 1, bool bottleneck = false,
                                      bool stride_on_3x3 = false) {
  if (input.dim() != 4 || input.size(3) <= 0)
    throw std::invalid_argument("positive NHWC ResNet input required");
  const std::int64_t in = input.size(3);
  if (in < 1 || width < 1 || (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid ResNet block dimensions");
  const std::int64_t out = width * (bottleneck ? 4 : 1);
  // Borrow this block prefix/stores and preserve caller BN mode for every branch layer.
  auto conv = [&](const std::string &part, const nn::Tensor &x, std::int64_t co,
                  std::int64_t kernel, int step, bool relu) {
    return ResNetConv(p, state, name + "/" + part, x, co, kernel, cfg, step, relu);
  };
  nn::Tensor h;
  if (bottleneck) {
    h = conv("conv1", input, width, 1, stride_on_3x3 ? 1 : stride, true);
    h = conv("conv2", h, width, 3, stride_on_3x3 ? stride : 1, true);
    h = conv("conv3", h, out, 1, 1, false);
  } else {
    h = conv("conv1", input, width, 3, stride, true);
    h = conv("conv2", h, out, 3, 1, false);
  }
  ResNetBlockOutputs result;
  result.residual = h;
  result.shortcut =
      in == out && stride == 1 ? input : conv("shortcut", input, out, 1, stride, false);
  result.preactivation = result.residual + result.shortcut;
  result.output = nn::relu(result.preactivation);
  return result;
}

/// @brief Reference ResNet depth, exact width scaling, stem, classifier and BN flags.
///
/// Value metadata only. Width scaling and the small-image stem are explicit
/// adaptations; all four stages and full block counts remain.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.small_stem = true;
/// cfg.Validate();
/// @endcode
/// @par References
/// - He et al. (2016), https://arxiv.org/abs/1512.03385; reference topology, https://github.com/KaimingHe/deep-residual-networks
/// @see ResNet()
struct ResNetConfig {
  int depth = 18;        ///< Learned depth: 18,34,50,101,152; projections excluded.
  int classes = 1000;    ///< At least two output scores, even when classifier disabled.
  int width_divisor = 1; ///< Positive exact divisor of 64; scales every stage width.
  bool small_stem =
      false; ///< True uses 3x3/stride-one/no pool; false uses the reference-sized stem.
  bool stride_on_3x3 =
      false; ///< True selects bottleneck v1.5 stride placement; ignored for basic blocks.
  double momentum = .9; ///< Finite BN old-value EMA retention in [0,1).
  bool include_classifier =
      true; ///< False creates no output-layer leaves and leaves logits undefined.
  bool update_stats =
      true; ///< False keeps moving state fixed even when using training batch moments.

  /// @brief Validate supported depth, class count, exact widths and BN momentum.
  ///
  /// @throws std::invalid_argument For unsupported depth or invalid class/width/momentum settings.
  /// @par Example
  /// @code
  /// dm::nn::models::ResNetConfig{}.Validate();
  /// @endcode
  /// @see ResNetBlockCounts()
  void Validate() const {
    ResNetBlockCounts(depth);
    if (classes < 2 || width_divisor < 1 || 64 % width_divisor ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid ResNet configuration");
  }
};
/// @brief Stem, four stage maps, every residual-block map and optional classifier scores.
///
/// Tensors retain storage/graphs. Model parameters and moving state belong to
/// the two caller stores; logits may be undefined when classifier is disabled.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.small_stem = true;
/// const auto out = n::models::ResNet(p, state, n::zeros({1, 32, 32, 3}), cfg);
/// @endcode
/// @see ResNet()
struct ResNetOutputs {
  nn::Tensor stem;   ///< Post-stem activation, after pooling unless small_stem.
  nn::Tensor logits; ///< Unnormalised [N,classes] scores, undefined if classifier omitted.
  std::array<nn::Tensor, 4> stages; ///< End of stages one through four.
  std::vector<nn::Tensor> blocks;   ///< Every post-add ReLU activation in execution order.
};

/// @brief Build a complete post-activation ResNet with explicit symmetric spatial padding.
///
/// The standard stem is 7x7/2 with pad three, then 3x3/2 max-pool with
/// negative-infinity pad one. Its 224-pixel sizes are 112 then 56; this explicit
/// edge alignment is not a bitwise Caffe-weights compatibility claim. Basic
/// depths 18/34 use two 3x3 layers; depths 50/101/152 use fourfold bottlenecks.
/// Stage sides halve between stages, then GAP feeds a linear classifier. Every
/// convolution is bias-free He initialized, BN learns gamma/beta with epsilon
/// 1e-5. Training uses eager unbiased-variance moving updates. No dropout,
/// pretrained weights, optimizer, smoothing or auxiliary losses are hidden here.
///
/// @param p Mutable floating learnable parameter store.
/// @param state Distinct mutable BN moving moments, matching p dtype.
/// @param image Nonempty floating NHWC RGB image batch, matching both stores.
/// @param cfg Valid depth, widths, stem/stride/classifier and BN settings.
/// @param training Use batch moments if true; update moving state only if cfg.update_stats.
/// @return Stem, all stage/block maps and optional logits; autograd history retained.
/// @throws std::invalid_argument For invalid cfg, RGB geometry, dtype, BN or stored parameter/state entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.small_stem = true;
/// const auto out = n::models::ResNet(p, state, n::zeros({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - He et al. (2016), https://arxiv.org/abs/1512.03385; reference topology, https://github.com/KaimingHe/deep-residual-networks
/// @see ResNetBlock(), ResNetConfig
inline ResNetOutputs ResNet(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const ResNetConfig &cfg = {},
                            bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(0) <= 0 || image.size(1) <= 0 || image.size(2) <= 0 ||
      image.size(3) != 3)
    throw std::invalid_argument("nonempty NHWC RGB ResNet input required");
  ResNetOutputs out;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = cfg.momentum;
  norm.epsilon = 1e-5;
  norm.update_stats = cfg.update_stats;
  auto h = ResNetConv(p, state, "stem", image, 64 / cfg.width_divisor, cfg.small_stem ? 3 : 7,
                      norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  const bool bottleneck = cfg.depth >= 50;
  const auto counts = ResNetBlockCounts(cfg.depth);
  for (int stage = 0; stage < 4; ++stage) {
    const int stage_stride = stage ? 2 : 1;
    const std::int64_t width = (64 << stage) / cfg.width_divisor;
    for (int block = 0; block < counts[stage]; ++block) {
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      h = ResNetBlock(p, state, name, h, width, norm, block == 0 ? stage_stride : 1,
                      bottleneck, cfg.stride_on_3x3)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[stage] = h;
  }
  if (cfg.include_classifier)
    out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
