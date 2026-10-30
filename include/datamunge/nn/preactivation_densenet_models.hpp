#pragma once
/// @file preactivation_densenet_models.hpp
/// @brief Preactivation residual and densely concatenated tensor vision models, without fitting.
#include "datamunge/nn/resnet_models.hpp"
#include <algorithm>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Learned-scale preactivation BN mode and eager moving-state settings.
///
/// Aggregate field order is training, update_stats, momentum, epsilon. These
/// settings are shared by preactivation residual and dense-concatenation layers.
/// No tensors or fitted state live in this configuration value.
///
/// @par Example
/// @code
/// dm::nn::models::PreactivationNormConfig norm;
/// norm.training = true;
/// norm.update_stats = false;
/// @endcode
/// @see PreactivationNormalize(), PreActivate()
struct PreactivationNormConfig {
  bool training = false;    ///< Use batch moments if true, stored moments otherwise.
  bool update_stats = true; ///< Update moving state only when training and true.
  double momentum = .9;     ///< Finite old-value EMA retention in [0,1).
  double epsilon = 1e-5;    ///< Finite strictly positive variance stabilizer.
};

/// @brief Normalize the last feature axis with learned gamma/beta, without applying ReLU.
///
/// Training uses batch population variance for output and unbiased variance for
/// eager EMA updates. Inference does not mutate state. No fitting or activation
/// is hidden in this helper. Shared stores are not safe for concurrent mutation.
///
/// @param p Mutable learnable floating store.
/// @param state Distinct mutable moving-moment store, same dtype as p.
/// @param name Prefix for learned gamma/beta and moving_mean/moving_var names.
/// @param input Nonempty floating tensor with batch/feature axes, matching store dtype.
/// @param cfg BN mode, moving update, momentum and epsilon settings.
/// @return Tensor shaped like input, retaining input and parameter gradients.
/// @throws std::invalid_argument For invalid input, BN settings, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreactivationNormConfig norm;
/// const auto y = n::models::PreactivationNormalize(p, state, "bn", n::ones({2, 4, 4, 3}), norm);
/// @endcode
/// @par References
/// - He et al. (2016), Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027; released CIFAR code, https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
/// @see PreActivate()
inline nn::Tensor PreactivationNormalize(nn::ParameterStore &p, nn::ParameterStore &state,
                                         const std::string &name, const nn::Tensor &input,
                                         const PreactivationNormConfig &cfg) {
  if (input.dim() < 2 || std::any_of(input.shape().begin(), input.shape().end(), [](auto d) {
        return d <= 0;
      }))
    throw std::invalid_argument("nonempty batch/feature tensor required");
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.update_stats = cfg.update_stats;
  return nn::batch_norm(p, state, name, input, cfg.training, bn);
}

/// @brief Apply learned-scale BN followed by ReLU before a weight layer.
///
/// Uses the same state semantics as PreactivationNormalize; no convolution or
/// optimizer step occurs here. Negative normalized entries are clipped.
///
/// @param p Mutable learnable floating store.
/// @param state Distinct mutable moving-moment store, same dtype as p.
/// @param name Prefix for learned gamma/beta and moving_mean/moving_var names.
/// @param input Nonempty floating tensor with batch/feature axes, matching store dtype.
/// @param cfg BN mode, moving update, momentum and epsilon settings.
/// @return Rectified normalized tensor with input.shape() and retained gradients.
/// @throws std::invalid_argument For invalid input, BN settings, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreactivationNormConfig norm;
/// const auto y = n::models::PreActivate(p, state, "bn", n::ones({2, 4, 4, 3}), norm);
/// @endcode
/// @par References
/// - He et al. (2016), Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027; released CIFAR code, https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
/// @see PreactivationNormalize(), PreActBottleneck()
inline nn::Tensor PreActivate(nn::ParameterStore &p, nn::ParameterStore &state,
                              const std::string &name, const nn::Tensor &input,
                              const PreactivationNormConfig &cfg) {
  return nn::relu(PreactivationNormalize(p, state, name, input, cfg));
}

/// @brief Apply a bias-free He convolution with symmetric floor(kernel/2) padding.
///
/// A single grouped HWIO tensor has input channels/groups and all output
/// channels. He-uniform fan-in initialization differs from the original
/// preactivation release's He-normal fan-out initialization. No fitting occurs.
///
/// @param p Mutable learnable store, creates/reuses name/W.
/// @param name Prefix for the sole HWIO kernel.
/// @param input Nonempty floating NHWC input, matching p dtype.
/// @param out Positive output channels, divisible by groups.
/// @param kernel Positive square kernel width; odd kernels preserve size at stride one.
/// @param stride Positive common spatial stride, default one.
/// @param groups Positive divisor of input/output channels, default one.
/// @return NHWC tensor retaining input/kernel gradients; no BN or activation.
/// @throws std::invalid_argument For invalid dimensions, groups, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::RawResidualConv(p, "c", n::ones({2, 5, 5, 4}), 6, 3, 2, 2);
/// @endcode
/// @par References
/// - He et al. (2016), Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027; released CIFAR code, https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
/// @see PreActivate(), PreActBottleneck()
inline nn::Tensor RawResidualConv(nn::ParameterStore &p, const std::string &name,
                                  const nn::Tensor &input, std::int64_t out,
                                  std::int64_t kernel, std::int64_t stride = 1,
                                  std::int64_t groups = 1) {
  if (input.dim() != 4 || input.size(0) <= 0 || input.size(1) <= 0 || input.size(2) <= 0 ||
      input.size(3) <= 0 || out <= 0 || kernel <= 0 || stride <= 0 || groups <= 0)
    throw std::invalid_argument("positive NHWC convolution dimensions/groups required");
  const std::int64_t in = input.size(3);
  if (in % groups || out % groups)
    throw std::invalid_argument("grouped convolution channels must divide by groups");
  const nn::Tensor w =
      p.get(name + "/W", {kernel, kernel, in / groups, out}, nn::init::he_uniform());
  nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Valid);
  o.groups = groups;
  if (kernel > 1) {
    o.padding = nn::Padding::Explicit;
    const std::int64_t pad = kernel / 2;
    o.explicit_paddings = {pad, pad, pad, pad};
  }
  return nn::conv2d(input, w, o);
}

/// @brief Apply a full preactivation bottleneck with an ungated sum after the shortcut.
///
/// Equal-width stride-one shortcuts carry raw signed input. Dimension-changing
/// projections consume the first BN/ReLU result, then use a raw 1x1 filter.
/// The branch is BN/ReLU/1x1, BN/ReLU/3x3, BN/ReLU/1x1. With zero branch, an
/// identity shortcut is exact on negative as well as positive values. No fitting.
///
/// @param p Mutable learnable floating store.
/// @param state Distinct moving-moment store with matching dtype.
/// @param name Block prefix; bn1/2/3, conv1/2/3 and optional shortcut names are appended.
/// @param input Nonempty floating NHWC feature map matching stores.
/// @param width Positive inner width; block emits 4*width channels.
/// @param stride One (default) or two, used in the first 1x1 and optional shortcut.
/// @param cfg Shared learned-scale BN settings, default PreactivationNormConfig.
/// @return Residual, shortcut and output=sum; preactivation aliases output, with no final ReLU.
/// @throws std::invalid_argument For invalid input, width, stride, BN or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreactivationNormConfig norm;
/// const auto y = n::models::PreActBottleneck(p, state, "b", n::ones({2, 5, 5, 4}), 1, 1, norm);
/// @endcode
/// @par References
/// - He et al. (2016), Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027; released CIFAR code, https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
/// @see PreActResNet(), ResNetBlockOutputs
inline ResNetBlockOutputs PreActBottleneck(nn::ParameterStore &p, nn::ParameterStore &state,
                                           const std::string &name, const nn::Tensor &input,
                                           std::int64_t width, int stride = 1,
                                           const PreactivationNormConfig &cfg = {}) {
  if (input.dim() != 4 || input.size(3) <= 0)
    throw std::invalid_argument("positive NHWC bottleneck input required");
  const std::int64_t in = input.size(3);
  if (in < 1 || width < 1 || (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid preactivation bottleneck");
  ResNetBlockOutputs result;
  const std::int64_t out = width * 4;
  nn::Tensor h = PreActivate(p, state, name + "/bn1", input, cfg);
  result.shortcut = in == out && stride == 1
                        ? input
                        : RawResidualConv(p, name + "/shortcut", h, out, 1, stride);
  h = RawResidualConv(p, name + "/conv1", h, width, 1, stride);
  h = PreActivate(p, state, name + "/bn2", h, cfg);
  h = RawResidualConv(p, name + "/conv2", h, width, 3);
  h = PreActivate(p, state, name + "/bn3", h, cfg);
  result.residual = RawResidualConv(p, name + "/conv3", h, out, 1);
  result.output = result.shortcut + result.residual;
  result.preactivation = result.output; // no post-addition nonlinearity
  return result;
}

/// @brief CIFAR 9n+2 bottleneck depth, base width, class count and BN update settings.
///
/// Reference depth 164 uses 18 bottlenecks per stage; 1001 uses 111. Width
/// scaling changes only channel widths; all three stages remain.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreActResNetConfig cfg;
/// cfg.depth = 11;
/// cfg.base_width = 4;
/// cfg.Validate();
/// @endcode
/// @par References
/// - He et al. (2016), Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027; released CIFAR code, https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
/// @see PreActResNet()
struct PreActResNetConfig {
  int depth = 164;          ///< At least 11 and of form 9n+2; three n-block bottleneck stages.
  int base_width = 16;      ///< Positive stem/first inner width; later stages double it.
  int classes = 10;         ///< At least two class logits.
  double momentum = .9;     ///< Finite old-value BN EMA retention in [0,1).
  bool update_stats = true; ///< Update moving moments during training only when true.

  /// @brief Validate CIFAR bottleneck depth, positive widths/classes and BN momentum.
  ///
  /// @throws std::invalid_argument For unsupported depth or invalid width/class/momentum values.
  /// @par Example
  /// @code
  /// dm::nn::models::PreActResNetConfig{}.Validate();
  /// @endcode
  /// @see PreActResNetConfig
  void Validate() const {
    if (depth < 11 || (depth - 2) % 9 || base_width < 1 || classes < 2 ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("preactivation CIFAR bottleneck depth must be 9n+2");
  }
};
/// @brief Raw stem, three ungated-sum stages, final preactivation and classifier scores.
///
/// Handles retain their tensor storage/graphs; fitted leaves and moments remain
/// in the two caller-owned stores.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreActResNetConfig cfg;
/// cfg.depth = 11;
/// cfg.base_width = 4;
/// const auto out = n::models::PreActResNet(p, state, n::zeros({1, 8, 8, 3}), cfg);
/// @endcode
/// @see PreActResNet()
struct PreActResNetOutputs {
  nn::Tensor stem;                  ///< Raw 3x3 stem convolution, without BN/ReLU.
  nn::Tensor features;              ///< Final learned BN/ReLU before global averaging.
  nn::Tensor logits;                ///< Unnormalised [N,classes] linear classifier scores.
  std::array<nn::Tensor, 3> stages; ///< End of each full bottleneck stage.
  std::vector<nn::Tensor> blocks;   ///< Every sum output, with no post-add activation.
};
/// @brief Build the authors' complete three-stage CIFAR preactivation bottleneck topology.
///
/// Raw 3x3 stem precedes three equally repeated stages. The first block of later
/// stages downsamples by two; outer widths are 4/8/16 times base_width. Final
/// BN/ReLU, GAP and a biased linear head follow. Learned-scale BN epsilon is
/// 1e-5. Initialization and eager unbiased-variance moving updates are documented
/// adaptations; no optimizer, dropout, pretrained weights or training schedule
/// is hidden here. All complete-depth activations are retained when recording
/// gradients, so deep models can consume substantial memory.
///
/// @param p Mutable floating learnable store.
/// @param state Distinct BN moving moments, matching p dtype.
/// @param image Nonempty floating NHWC RGB batch.
/// @param cfg Valid 9n+2 depth, base width, classes and moving update flags.
/// @param training Use batch moments and optional eager updates when true; inference by default.
/// @return Stem, all block/stage maps, final normalized features and [N,classes] logits.
/// @throws std::invalid_argument For invalid cfg, image geometry, BN, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreActResNetConfig cfg;
/// cfg.depth = 11;
/// cfg.base_width = 4;
/// const auto out = n::models::PreActResNet(p, state, n::zeros({1, 8, 8, 3}), cfg);
/// @endcode
/// @par References
/// - He et al. (2016), Identity Mappings in Deep Residual Networks, https://arxiv.org/abs/1603.05027; released CIFAR code, https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
/// @see PreActBottleneck(), PreActResNetConfig
inline PreActResNetOutputs PreActResNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                        const nn::Tensor &image,
                                        const PreActResNetConfig &cfg = {},
                                        bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(0) <= 0 || image.size(1) <= 0 || image.size(2) <= 0 ||
      image.size(3) != 3)
    throw std::invalid_argument("nonempty NHWC RGB image required");
  PreActResNetOutputs out;
  const PreactivationNormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  const int repetitions = (cfg.depth - 2) / 9;
  nn::Tensor h = RawResidualConv(p, "stem", image, cfg.base_width, 3);
  out.stem = h;
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t width = std::int64_t(cfg.base_width) << stage;
    for (int block = 0; block < repetitions; ++block) {
      h = PreActBottleneck(p, state,
                           "stage" + std::to_string(stage + 1) + "/block" +
                               std::to_string(block + 1),
                           h, width, stage > 0 && block == 0 ? 2 : 1, norm)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[stage] = h;
  }
  out.features = PreActivate(p, state, "final_bn", h, norm);
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}
/// @brief New feature channels and the unchanged-input/new-feature concatenation.
///
/// Tensor handles own storage references and retain their gradient graphs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto out = n::models::DenseGrowthLayer(p, state, "g", n::ones({2, 5, 5, 3}), 2);
/// @endcode
/// @see DenseGrowthLayer()
struct DenseGrowthOutputs {
  nn::Tensor
      output; ///< NHWC concatenation [input,added], preserving every old channel verbatim.
  nn::Tensor added; ///< The growth new channels, possibly signed after the raw final filter.
};
/// @brief Append growth new channels using BN/ReLU and an optional 4*growth bottleneck.
///
/// Raw final convolution adds no output ReLU. Direct derivatives through the old
/// channel slice are exactly identity; losses on added channels also contribute
/// indirect input gradients. Work per layer is O(NHW 9 Cin growth) without
/// bottleneck, or O(NHW (4 Cin growth+36 growth²)) with one, excluding BN.
/// Concatenation is materialized, rather than using shared buffers/recomputation.
///
/// @param p Mutable learnable floating store.
/// @param state Distinct BN moving moments, matching p dtype.
/// @param name Prefix for bn1/conv1 and optional bn2, plus final conv2.
/// @param input Nonempty floating NHWC map; all channels retained verbatim.
/// @param growth Positive number of new channels.
/// @param bottleneck Use preactivated 1x1 to 4*growth then preactivated 3x3 if true (default); otherwise one 3x3.
/// @param norm Preactivation BN settings; default inference with learned gamma/beta.
/// @return added has growth channels; output=[input,added] has Cin+growth, same spatial grid.
/// @throws std::invalid_argument For invalid input/growth, BN, dtype or store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto out = n::models::DenseGrowthLayer(p, state, "g", n::ones({2, 5, 5, 3}), 2);
/// @endcode
/// @par References
/// - Huang et al. (2017), Densely Connected Convolutional Networks, https://arxiv.org/abs/1608.06993; released model, https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
/// @see DenseTransition(), DenseNet()
inline DenseGrowthOutputs DenseGrowthLayer(nn::ParameterStore &p, nn::ParameterStore &state,
                                           const std::string &name, const nn::Tensor &input,
                                           std::int64_t growth, bool bottleneck = true,
                                           const PreactivationNormConfig &norm = {}) {
  if (input.dim() != 4 || input.size(3) <= 0)
    throw std::invalid_argument("positive NHWC dense input required");
  const std::int64_t in = input.size(3);
  if (in < 1 || growth < 1)
    throw std::invalid_argument("positive DenseNet channel counts required");
  nn::Tensor h = PreActivate(p, state, name + "/bn1", input, norm);
  if (bottleneck) {
    h = RawResidualConv(p, name + "/conv1", h, 4 * growth, 1);
    h = PreActivate(p, state, name + "/bn2", h, norm);
  }
  DenseGrowthOutputs out;
  out.added = RawResidualConv(p, name + "/conv2", h, growth, 3);
  out.output = nn::concat({input, out.added}, 3);
  return out;
}

/// @brief Compress channels with BN/ReLU/1x1, then VALID 2x2 average-pool by two.
///
/// The 1x1 is bias-free and has no output activation. Pooling averages four
/// spatial values and drops incomplete last rows/columns. No fitting occurs.
/// The pointwise kernel costs Cin*out weights plus BN gamma/beta on Cin.
///
/// @param p Mutable learnable floating store.
/// @param state Distinct BN moving moments, matching p dtype.
/// @param name Prefix for /bn and /conv/W.
/// @param input Floating NHWC map with H,W at least two and positive channels.
/// @param out Positive retained channel count, at most Cin.
/// @param norm Preactivation BN settings; defaults to inference.
/// @return [N,floor(H/2),floor(W/2),out] floating map retaining its gradient graph.
/// @throws std::invalid_argument For invalid compression, input geometry, BN, dtype or store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::DenseTransition(p, state, "t", n::ones({2, 4, 4, 8}), 4);
/// @endcode
/// @par References
/// - Huang et al. (2017), Densely Connected Convolutional Networks, https://arxiv.org/abs/1608.06993; released model, https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
/// @see DenseGrowthLayer(), DenseNet()
inline nn::Tensor DenseTransition(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const std::string &name, const nn::Tensor &input,
                                  std::int64_t out, const PreactivationNormConfig &norm = {}) {
  if (input.dim() != 4 || input.size(1) < 2 || input.size(2) < 2 || input.size(3) <= 0)
    throw std::invalid_argument("dense transition requires NHWC with H,W>=2");
  const std::int64_t in = input.size(3);
  if (in < 1 || out < 1 || out > in)
    throw std::invalid_argument(
        "DenseNet compression must retain between one and all channels");
  nn::Tensor h = PreActivate(p, state, name + "/bn", input, norm);
  h = RawResidualConv(p, name + "/conv", h, out, 1);
  return nn::avg_pool2d(h, 2, 2, nn::Padding::Valid);
}

/// @brief ImageNet/CIFAR DenseNet depth, growth, bottlenecks and transition compression.
///
/// The initial channel width is twice growth_rate. Compression floors retained
/// width after each block; it may not eliminate all channels. ImageNet presets
/// select block counts, not a growth rate: the released DenseNet-161 uses growth
/// 48, while the default is 32. No learned tensors live in this value.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DenseNetConfig cfg;
/// cfg.cifar_layout = true;
/// cfg.depth = 10;
/// cfg.growth_rate = 2;
/// cfg.classes = 3;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Huang et al. (2017), Densely Connected Convolutional Networks, https://arxiv.org/abs/1608.06993; released model, https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
/// @see DenseNet()
struct DenseNetConfig {
  int depth = 121; ///< ImageNet preset or CIFAR 6n+4 (bottleneck) / 3n+4 (basic).
  int growth_rate =
      32; ///< Positive new channels appended per dense layer; stem width=2*growth_rate.
  int classes = 1000; ///< At least two output logits.
  bool cifar_layout =
      false; ///< Three small-image stages if true, four ImageNet stages otherwise.
  bool bottleneck = true;   ///< Use 4k reductions if true; mandatory for ImageNet presets.
  double compression = .5;  ///< Retained transition fraction in (0,1], rounded down.
  double momentum = .9;     ///< Finite BN old-value EMA retention in [0,1).
  bool update_stats = true; ///< Update moving moments only on training calls when true.

  /// @brief Resolve three CIFAR repetition counts or the four ImageNet block counts.
  ///
  /// ImageNet supports 121/169/201/161; CIFAR depth is 6n+4 with bottlenecks
  /// or 3n+4 without. Growth/compression settings are checked by Validate().
  ///
  /// @return Positive dense-layer counts, without mutating this configuration.
  /// @throws std::invalid_argument For unsupported depth/layout or ImageNet without bottlenecks.
  /// @par Example
  /// @code
  /// const auto counts = dm::nn::models::DenseNetConfig{}.BlockCounts(); // 6,12,24,16
  /// @endcode
  /// @see Validate()
  std::vector<int> BlockCounts() const {
    if (cifar_layout) {
      const int divisor = bottleneck ? 6 : 3;
      if (depth <= 4 || (depth - 4) % divisor)
        throw std::invalid_argument("CIFAR DenseNet depth must be 6n+4 (bottleneck) or 3n+4");
      return std::vector<int>(3, (depth - 4) / divisor);
    }
    if (!bottleneck)
      throw std::invalid_argument("ImageNet presets use bottleneck DenseNet");
    switch (depth) {
    case 121:
      return {6, 12, 24, 16};
    case 169:
      return {6, 12, 32, 32};
    case 201:
      return {6, 12, 48, 32};
    case 161:
      return {6, 12, 36, 24};
    default:
      throw std::invalid_argument("ImageNet DenseNet depth must be 121/169/201/161");
    }
  }
  /// @brief Validate depth, growth, class count, BN momentum and nonempty compressed widths.
  ///
  /// @throws std::invalid_argument For invalid settings or a transition that floors to zero retained channels.
  /// @par Example
  /// @code
  /// dm::nn::models::DenseNetConfig{}.Validate();
  /// @endcode
  /// @see BlockCounts()
  void Validate() const {
    BlockCounts();
    if (growth_rate < 1 || classes < 2 || !(compression > 0 && compression <= 1) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid DenseNet configuration");
    std::int64_t channels = 2LL * growth_rate;
    const auto blocks = BlockCounts();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      channels += std::int64_t(blocks[i]) * growth_rate;
      if (i + 1 < blocks.size()) {
        channels = static_cast<std::int64_t>(std::floor(double(channels) * compression));
        if (channels < 1)
          throw std::invalid_argument("DenseNet compression eliminates all channels");
      }
    }
  }
};
/// @brief DenseNet stem, every growing map, block/transition maps and classifier outputs.
///
/// Handles retain their storage/graphs; concatenated maps are materialized.
/// Parameters and moving moments remain in the caller-owned stores.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DenseNetConfig cfg;
/// cfg.cifar_layout = true;
/// cfg.depth = 10;
/// cfg.growth_rate = 2;
/// cfg.classes = 3;
/// const auto out = n::models::DenseNet(p, state, n::zeros({1, 8, 8, 3}), cfg);
/// @endcode
/// @see DenseNet()
struct DenseNetOutputs {
  nn::Tensor stem; ///< Initial 2k-channel map, after BN/ReLU/pool only for ImageNet layout.
  nn::Tensor features;                 ///< Final learned BN/ReLU map before global mean.
  nn::Tensor logits;                   ///< Unnormalised [N,classes] linear scores.
  std::vector<nn::Tensor> stages;      ///< End of each dense block.
  std::vector<nn::Tensor> transitions; ///< Compressed, downsampled maps between blocks.
  std::vector<nn::Tensor> layers;      ///< Every growing concatenation in execution order.
};
/// @brief Build the complete selected DenseNet topology without fitting parameters.
///
/// The ImageNet stem uses 7x7/2 and BN/ReLU/padded max-pool; CIFAR uses raw
/// 3x3/1. Each layer appends growth_rate channels. Transitions floor compression
/// then halve each spatial side. Final BN/ReLU, GAP and biased linear scores
/// follow. BN learns gamma/beta with epsilon 1e-5 and unbiased EMA variance.
/// He-uniform initialization and eager state updates are adaptations. No
/// dropout, pretrained weights, optimizer or activation recomputation is hidden.
/// Materializing each concatenation incurs O(NHW(L C0+k L²)) accumulated
/// feature storage in a recorded L-layer dense block.
///
/// @param p Mutable learnable floating store.
/// @param state Distinct BN moving-moment store, matching p dtype.
/// @param image Nonempty floating NHWC RGB input, spatially large enough for all VALID transition pools.
/// @param cfg Valid layout, depth, growth, bottleneck/compression and BN settings.
/// @param training Use batch moments and optional eager moving-state updates if true; false by default.
/// @return All feature maps and [N,classes] scores, retaining their differentiation graphs.
/// @throws std::invalid_argument For invalid cfg, RGB/transition geometry, BN, dtype or store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DenseNetConfig cfg;
/// cfg.cifar_layout = true;
/// cfg.depth = 10;
/// cfg.growth_rate = 2;
/// cfg.classes = 3;
/// const auto out = n::models::DenseNet(p, state, n::zeros({1, 8, 8, 3}), cfg);
/// @endcode
/// @par References
/// - Huang et al. (2017), Densely Connected Convolutional Networks, https://arxiv.org/abs/1608.06993; released model, https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
/// @see DenseGrowthLayer(), DenseTransition()
inline DenseNetOutputs DenseNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                const nn::Tensor &image, const DenseNetConfig &cfg = {},
                                bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(0) <= 0 || image.size(1) <= 0 || image.size(2) <= 0 ||
      image.size(3) != 3)
    throw std::invalid_argument("nonempty NHWC RGB image required");
  DenseNetOutputs out;
  const PreactivationNormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  std::int64_t channels = 2 * cfg.growth_rate;
  nn::Tensor h = RawResidualConv(p, "stem", image, channels, cfg.cifar_layout ? 3 : 7,
                                 cfg.cifar_layout ? 1 : 2);
  if (!cfg.cifar_layout) {
    h = PreActivate(p, state, "stem_bn", h, norm);
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  }
  out.stem = h;
  const auto counts = cfg.BlockCounts();
  for (std::size_t stage = 0; stage < counts.size(); ++stage) {
    for (int layer = 0; layer < counts[stage]; ++layer) {
      h = DenseGrowthLayer(p, state,
                           "stage" + std::to_string(stage + 1) + "/layer" +
                               std::to_string(layer + 1),
                           h, cfg.growth_rate, cfg.bottleneck, norm)
              .output;
      channels += cfg.growth_rate;
      out.layers.push_back(h);
    }
    out.stages.push_back(h);
    if (stage + 1 < counts.size()) {
      const auto compressed =
          static_cast<std::int64_t>(std::floor(double(channels) * cfg.compression));
      h = DenseTransition(p, state, "transition" + std::to_string(stage + 1), h, compressed,
                          norm);
      channels = compressed;
      out.transitions.push_back(h);
    }
  }
  out.features = PreActivate(p, state, "final_bn", h, norm);
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
