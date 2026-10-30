#pragma once
/// @file grouped_wide_residual_models.hpp
/// @brief Grouped ResNeXt and preactivated Wide ResNet tensor forward models, without fitting.
#include "datamunge/nn/preactivation_densenet_models.hpp"
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Apply a Form-C grouped bottleneck followed by shortcut addition and ReLU.
///
/// Middle width is cardinality*floor(planes*base_width/64). Reduction/expansion
/// are dense 1x1, with learned-scale BN when cfg.scale=true. The grouped kernel
/// is one HWIO leaf [3,3,middle/cardinality,middle]; no convolution bias.
/// Matching width/stride uses raw identity. No optimizer or preprocessing is hidden.
/// He-uniform fan-in replaces the authors normal fan-out initialization.
/// Work O(NHW*(Cin*middle+9*middle²/cardinality+4*planes*middle)).
///
/// @param p Mutable learnable store; lazily creates/reuses named leaves.
/// @param state Mutable BN moving-moment store; updates only when cfg requests training updates.
/// @param name Prefix for conv1, conv2, conv3 and optional shortcut.
/// @param input Nonempty floating NHWC tensor matching store dtype.
/// @param planes Positive stage base width; output has 4*planes channels.
/// @param cardinality Positive number of independent contiguous 3x3 channel groups.
/// @param base_width Positive group-width coefficient; floor(planes*base_width/64) must be positive.
/// @param cfg BN/ReLU settings, including training, scale, momentum and epsilon.
/// @param stride One or two, applied in the grouped 3x3 and projection.
/// @return Residual, shortcut, preactivation sum and post-ReLU output with gradients.
/// @throws std::invalid_argument For invalid input, widths, stride, BN options, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig norm;
/// norm.scale = true;
/// const auto y = n::models::ResNeXtBlock(p, state, "b", n::ones({1, 5, 5, 8}), 4, 2, 16, norm);
/// @endcode
/// @par References
/// - Xie et al. (2017), https://arxiv.org/abs/1611.05431; authors Form C: https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
/// @see ResNeXt(), ResNetBlockOutputs
inline ResNetBlockOutputs ResNeXtBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const std::string &name, const nn::Tensor &input,
                                       std::int64_t planes, std::int64_t cardinality,
                                       std::int64_t base_width, const ResNetNormConfig &cfg,
                                       int stride = 1) {
  if (input.dim() != 4 || input.numel() == 0)
    throw std::invalid_argument("ResNeXt input must be nonempty NHWC");
  const std::int64_t in = input.shape().back(), group_width = planes * base_width / 64,
                     middle = group_width * cardinality;
  const std::int64_t out = 4 * planes;
  if (in < 1 || planes < 1 || cardinality < 1 || base_width < 1 || group_width < 1 ||
      (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid ResNeXt bottleneck dimensions");
  ResNetBlockOutputs result;
  auto h = ResNetConv(p, state, name + "/conv1", input, middle, 1, cfg);
  h = RawResidualConv(p, name + "/conv2", h, middle, 3, stride, cardinality);
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.scale = cfg.scale;
  bn.update_stats = cfg.update_stats;
  h = nn::relu(nn::batch_norm(p, state, name + "/conv2/bn", h, cfg.training, bn));
  result.residual = ResNetConv(p, state, name + "/conv3", h, out, 1, cfg, 1, false);
  result.shortcut = in == out && stride == 1 ? input
                                             : ResNetConv(p, state, name + "/shortcut", input,
                                                          out, 1, cfg, stride, false);
  result.preactivation = result.shortcut + result.residual;
  result.output = nn::relu(result.preactivation);
  return result;
}

/// @brief Depth, cardinality and group width for ImageNet or CIFAR ResNeXt.
///
/// Value metadata only; defaults describe ImageNet ResNeXt-50 32x4d.
/// CIFAR uses 9n+2 depth and three stages; compact dimensions are explicit.
///
/// @par Example
/// @code
/// dm::nn::models::ResNeXtConfig cfg;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Xie et al. (2017), https://arxiv.org/abs/1611.05431; authors Form C: https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
/// @see ResNeXt()
struct ResNeXtConfig {
  int depth = 50;            ///< ImageNet 50/101/152, or CIFAR 9n+2 with n>=1.
  int cardinality = 32;      ///< Positive grouped 3x3 cardinality C.
  int base_width = 4;        ///< Positive group width at stage base 64, before flooring.
  int stage_width = 64;      ///< Positive initial stage base/stem width.
  int classes = 1000;        ///< Classifier output width, at least two.
  bool cifar_layout = false; ///< Three small-image stages and a 3x3 stride-one stem.
  double momentum = .9;      ///< Moving BN retention in [0,1).
  bool update_stats = true;  ///< Allow moving-moment updates in training mode.

  /// @brief Validate ResNeXt layout, dimensions and probability settings.
  ///
  /// Does not allocate or mutate learned state.
  ///
  /// @throws std::invalid_argument For invalid depth, width, classes, momentum or cardinality.
  /// @par Example
  /// @code
  /// dm::nn::models::ResNeXtConfig{}.Validate();
  /// @endcode
  /// @see ResNeXtConfig
  void Validate() const {
    if (cifar_layout ? (depth < 11 || (depth - 2) % 9)
                     : (depth != 50 && depth != 101 && depth != 152))
      throw std::invalid_argument(
          "ResNeXt requires ImageNet depth 50/101/152 or CIFAR depth 9n+2");
    if (cardinality < 1 || base_width < 1 || stage_width < 1 ||
        stage_width * base_width < 64 || classes < 2 || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid ResNeXt configuration");
  }
};
/// @brief Stem, stage activations, ordered blocks and classifier outputs from ResNeXt.
///
/// Handles own storage references and retain autograd graphs; stores own learned
/// leaves and moving moments. Activations are not detached or copied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNeXtConfig cfg;
/// cfg.depth = 11;
/// cfg.cifar_layout = true;
/// cfg.stage_width = 16;
/// cfg.cardinality = 2;
/// cfg.base_width = 16;
/// cfg.classes = 10;
/// const auto y = n::models::ResNeXt(p, state, n::zeros({1, 16, 16, 3}), cfg);
/// @endcode
/// @see ResNeXt()
struct ResNeXtOutputs {
  nn::Tensor stem;   ///< Post-BN/ReLU stem, including pool for ImageNet.
  nn::Tensor logits; ///< Unnormalised [N,classes] scores after global average pooling.
  std::vector<nn::Tensor> stages; ///< Three CIFAR or four ImageNet stage ends.
  std::vector<nn::Tensor> blocks; ///< Every post-addition ReLU in execution order.
};
/// @brief Build the complete ResNeXt tensor classifier without fitting.
///
/// ImageNet has a 7x7 stride-two stem, padded 3x3 stride-two max-pool and
/// four 3/4/6/3 stages at depth 50. CIFAR uses 3x3 stride one, no initial
/// pool and three (depth-2)/9 stages. Stage outer widths are 4*stage_width*2^s.
/// Stride two is in the grouped 3x3 at later stage entries; final global mean
/// accepts variable spatial dimensions. Full 50 32x4d has 25028904 learned
/// scalars. Learned-scale BN epsilon 1e-5, no dropout.
/// No input preprocessing, loss, optimizer or fitting is hidden. Parameters
/// use He-uniform fan-in convolutions, with a Glorot dense classifier.
/// Moving moments update only when training && cfg.update_stats. Work sums
/// convolution MACs; storage includes intermediate activations and learned leaves.
///
/// @param p Mutable learnable floating store; names are reused on repeated calls.
/// @param state Mutable moving-moment store, matching p dtype.
/// @param image Nonempty floating NHWC image tensor with positive channels.
/// @param cfg Validated ResNeXt configuration; default ImageNet 50 32x4d.
/// @param training Use batch moments and training dropout when applicable; false uses saved moments.
/// @return Stem, stages, every block and logits, all retaining gradients.
/// @throws std::invalid_argument For invalid configuration, NHWC, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNeXtConfig cfg;
/// cfg.depth = 11;
/// cfg.cifar_layout = true;
/// cfg.stage_width = 16;
/// cfg.cardinality = 2;
/// cfg.base_width = 16;
/// cfg.classes = 10;
/// const auto y = n::models::ResNeXt(p, state, n::zeros({1, 16, 16, 3}), cfg);
/// @endcode
/// @par References
/// - Xie et al. (2017), https://arxiv.org/abs/1611.05431; authors Form C: https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
/// @see ResNeXtConfig, ResNeXtBlock()
inline ResNeXtOutputs ResNeXt(nn::ParameterStore &p, nn::ParameterStore &state,
                              const nn::Tensor &image, const ResNeXtConfig &cfg = {},
                              bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.numel() == 0)
    throw std::invalid_argument("model image must be nonempty NHWC");
  ResNeXtOutputs out;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.epsilon = 1e-5;
  norm.momentum = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  auto h = ResNetConv(p, state, "stem", image, cfg.stage_width, cfg.cifar_layout ? 3 : 7, norm,
                      cfg.cifar_layout ? 1 : 2);
  if (!cfg.cifar_layout)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  const int r = (cfg.depth - 2) / 9;
  const auto counts =
      cfg.cifar_layout ? std::array<int, 4>{r, r, r, 0} : ResNetBlockCounts(cfg.depth);
  for (int stage = 0; stage < (cfg.cifar_layout ? 3 : 4); ++stage) {
    const std::int64_t planes = std::int64_t(cfg.stage_width) << stage;
    for (int block = 0; block < counts[static_cast<std::size_t>(stage)]; ++block) {
      h = ResNeXtBlock(
              p, state,
              "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h,
              planes, cfg.cardinality, cfg.base_width, norm, stage > 0 && block == 0 ? 2 : 1)
              .output;
      out.blocks.push_back(h);
    }
    out.stages.push_back(h);
  }
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}
/// @brief Wide-block paths and activations surrounding its branch-only dropout.
///
/// Tensor handles retain storage and differentiation graphs; no model state is copied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::Generator gen(17);
/// n::models::PreactivationNormConfig norm;
/// const auto y =
///     n::models::WideBasicBlock(p, state, "b", n::ones({1, 5, 5, 4}), 4, 1, norm, .3, gen);
/// @endcode
/// @see WideBasicBlock()
struct WideBlockOutputs {
  nn::Tensor residual;       ///< Second 3x3 convolution, without post-addition activation.
  nn::Tensor shortcut;       ///< Raw identity or preactivated 1x1 projection.
  nn::Tensor output;         ///< shortcut+residual, retaining negative values.
  nn::Tensor before_dropout; ///< Second BN/ReLU activation.
  nn::Tensor after_dropout;  ///< Inverted elementwise mask in training, otherwise unchanged.
};
/// @brief Apply two preactivated 3x3 convolutions with optional branch-only dropout.
///
/// Matching dimensions carry the raw input; learned projections consume the first
/// BN/ReLU. Dropout follows the second BN/ReLU and precedes conv2; shortcuts
/// are never dropped. Retained values/derivatives scale by 1/(1-dropout).
/// Inference does not consume gen. No convolution biases or post-addition ReLU.
/// He-uniform fan-in weights; no fitting. Work O(NHW*9*(Cin*out+out²)).
///
/// @param p Mutable learnable store, lazily creates/reuses leaves.
/// @param state BN moving moments, updated according to norm.
/// @param name Prefix for bn1, shortcut, conv1, bn2 and conv2.
/// @param input Nonempty floating NHWC tensor matching store dtype.
/// @param out Positive output channel width.
/// @param stride One or two, applied in first convolution and projection.
/// @param norm Preactivation settings; epsilon normally 1e-5.
/// @param dropout Elementwise removal probability in [0,1), training only.
/// @param gen Caller-owned generator, advanced only when dropout is sampled.
/// @return Branch and shortcut tensors, signed sum, and pre/post-dropout activations.
/// @throws std::invalid_argument For invalid shape, stride, dropout, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::Generator gen(17);
/// n::models::PreactivationNormConfig norm;
/// const auto y =
///     n::models::WideBasicBlock(p, state, "b", n::ones({1, 5, 5, 4}), 4, 1, norm, .3, gen);
/// @endcode
/// @par References
/// - Zagoruyko and Komodakis (2016), https://arxiv.org/abs/1605.07146; https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
/// @see WideResNet(), WideBlockOutputs
inline WideBlockOutputs WideBasicBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const std::string &name, const nn::Tensor &input,
                                       std::int64_t out, int stride,
                                       const PreactivationNormConfig &norm, double dropout,
                                       nn::Generator &gen) {
  if (input.dim() != 4 || input.numel() == 0)
    throw std::invalid_argument("wide block input must be nonempty NHWC");
  const std::int64_t in = input.shape().back();
  if (in < 1 || out < 1 || (stride != 1 && stride != 2) || !(dropout >= 0 && dropout < 1))
    throw std::invalid_argument("invalid wide residual block");
  WideBlockOutputs result;
  auto h = PreActivate(p, state, name + "/bn1", input, norm);
  result.shortcut = in == out && stride == 1
                        ? input
                        : RawResidualConv(p, name + "/shortcut", h, out, 1, stride);
  h = RawResidualConv(p, name + "/conv1", h, out, 3, stride);
  result.before_dropout = PreActivate(p, state, name + "/bn2", h, norm);
  result.after_dropout = norm.training && dropout > 0
                             ? nn::dropout(result.before_dropout, dropout, true, gen)
                             : result.before_dropout;
  result.residual = RawResidualConv(p, name + "/conv2", result.after_dropout, out, 3);
  result.output = result.shortcut + result.residual;
  return result;
}

/// @brief CIFAR Wide ResNet depth, widening and residual-dropout settings.
///
/// Defaults describe WRN-28-10 with dropout .3 and ten output classes.
/// Configuration stores metadata, without allocating or fitting parameters.
///
/// @par Example
/// @code
/// dm::nn::models::WideResNetConfig cfg;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Zagoruyko and Komodakis (2016), https://arxiv.org/abs/1605.07146; https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
/// @see WideResNet()
struct WideResNetConfig {
  int depth = 28;        ///< Depth 6n+4 with n>=1; excludes shortcut projection convolutions.
  int widen_factor = 10; ///< Positive multiplier k for stage widths 16k/32k/64k.
  int classes = 10;      ///< Output classes, at least two.
  double dropout = .3;   ///< Branch elementwise removal probability in [0,1).
  double momentum = .9;  ///< BN moving retention in [0,1).
  bool update_stats = true; ///< Permit moving-moment updates during training.

  /// @brief Validate WideResNet layout, dimensions and probability settings.
  ///
  /// Does not allocate or mutate learned state.
  ///
  /// @throws std::invalid_argument For invalid depth, width, classes, momentum or dropout.
  /// @par Example
  /// @code
  /// dm::nn::models::WideResNetConfig{}.Validate();
  /// @endcode
  /// @see WideResNetConfig
  void Validate() const {
    if (depth < 10 || (depth - 4) % 6 || widen_factor < 1 || classes < 2 ||
        !(dropout >= 0 && dropout < 1) || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument(
          "wide residual depth must be 6n+4, with positive width and valid dropout");
  }
};
/// @brief Stem, stage activations, ordered blocks and classifier outputs from WideResNet.
///
/// Handles own storage references and retain autograd graphs; stores own learned
/// leaves and moving moments. Activations are not detached or copied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::Generator gen(17);
/// n::models::WideResNetConfig cfg;
/// cfg.depth = 16;
/// cfg.widen_factor = 1;
/// const auto y = n::models::WideResNet(p, state, n::zeros({1, 16, 16, 3}), cfg, false, gen);
/// @endcode
/// @see WideResNet()
struct WideResNetOutputs {
  nn::Tensor stem;     ///< Fixed 16-channel 3x3 stride-one stem.
  nn::Tensor features; ///< Final BN/ReLU after the third stage.
  nn::Tensor logits;   ///< Unnormalised [N,classes] global-pool/dense scores.
  std::array<nn::Tensor, 3>
      stages;                     ///< Signed stage ends at input, half and quarter resolution.
  std::vector<nn::Tensor> blocks; ///< Ordered (depth-4)/2 residual-unit sums.
};
/// @brief Build the complete WideResNet tensor classifier without fitting.
///
/// Fixed 16-channel 3x3 stem; three stages contain (depth-4)/6 units each
/// at 16k/32k/64k channels, downsampling at stage 2/3 entries. Final BN/ReLU,
/// global mean and dense classifier. Full WRN-28-10 has 36479194 learned
/// scalars. Learned-scale BN epsilon 1e-5; dropout only inside branches.
/// One caller-owned random stream replaces release per-block streams.
/// No input preprocessing, loss, optimizer or fitting is hidden. Parameters
/// use He-uniform fan-in convolutions, with a Glorot dense classifier.
/// Moving moments update only when training && cfg.update_stats. Work sums
/// convolution MACs; storage includes intermediate activations and learned leaves.
///
/// @param p Mutable learnable floating store; names are reused on repeated calls.
/// @param state Mutable moving-moment store, matching p dtype.
/// @param image Nonempty floating NHWC image tensor with positive channels.
/// @param cfg Validated WideResNet configuration.
/// @param training Use batch moments and training dropout when applicable; false uses saved moments.
/// @param gen Caller-owned dropout generator, advanced in training when cfg.dropout>0.
/// @return Stem, stages, every block and logits, all retaining gradients.
/// @throws std::invalid_argument For invalid configuration, NHWC, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::Generator gen(17);
/// n::models::WideResNetConfig cfg;
/// cfg.depth = 16;
/// cfg.widen_factor = 1;
/// const auto y = n::models::WideResNet(p, state, n::zeros({1, 16, 16, 3}), cfg, false, gen);
/// @endcode
/// @par References
/// - Zagoruyko and Komodakis (2016), https://arxiv.org/abs/1605.07146; https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
/// @see WideResNetConfig, WideBasicBlock()
inline WideResNetOutputs WideResNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &image, const WideResNetConfig &cfg,
                                    bool training, nn::Generator &gen) {
  cfg.Validate();
  if (image.dim() != 4 || image.numel() == 0)
    throw std::invalid_argument("model image must be nonempty NHWC");
  WideResNetOutputs out;
  const PreactivationNormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  nn::Tensor h = RawResidualConv(p, "stem", image, 16, 3);
  out.stem = h;
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t channels = (std::int64_t(16) << stage) * cfg.widen_factor;
    for (int block = 0; block < (cfg.depth - 4) / 6; ++block) {
      h = WideBasicBlock(p, state,
                         "stage" + std::to_string(stage + 1) + "/block" +
                             std::to_string(block + 1),
                         h, channels, stage > 0 && block == 0 ? 2 : 1, norm, cfg.dropout, gen)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[static_cast<std::size_t>(stage)] = h;
  }
  out.features = PreActivate(p, state, "final_bn", h, norm);
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
