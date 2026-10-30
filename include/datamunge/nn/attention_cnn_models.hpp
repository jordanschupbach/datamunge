#pragma once
/// @file attention_cnn_models.hpp
/// @brief Tensor attention/deformable modules and complete residual classifiers; no fitting.
#include "datamunge/nn/deformable_models.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief A differentiable channel gate and its rescaled feature map.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned/state leaves remain owned by their stores; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::Eca(p, "e", n::ones({1, 4, 4, 8}), 3);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNet(), SqueezeExcitationModule(), Cbam(), Eca(), SelectiveKernel()
struct GateOutputs {
  nn::Tensor gate;   ///< Logistic [N,1,1,C] channel weights.
  nn::Tensor output; ///< Input multiplied by gate, retaining full gate/input derivatives.
};

/// @brief Apply global-mean squeeze and two biased pointwise channel recalibration layers.
///
/// Global mean -> biased He-uniform 1x1/ReLU -> biased 1x1/sigmoid.
/// No normalization. Work O(NHW*C+N*C*floor(C/r)); learned count
/// is 2*C*floor(C/r)+floor(C/r)+C. Gate/input derivatives both retained.
///
/// @param p Mutable floating learned store, lazily creates/reuses leaves.
/// @param name Prefix for /reduce and /expand kernels/biases.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @param reduction Positive integer at most C; hidden width floor(C/reduction).
/// @return Logistic gate and rescaled NHWC output.
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::SqueezeExcitationModule(p, "se", n::ones({2, 5, 5, 8}), 2);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline GateOutputs SqueezeExcitationModule(nn::ParameterStore &p, const std::string &name,
                                           const nn::Tensor &x, std::int64_t reduction) {
  if (x.dim() != 4 || x.numel() == 0 || reduction < 1 || reduction > x.size(3))
    throw std::invalid_argument("invalid SE dimensions/reduction");
  const std::int64_t channels = x.shape()[3];
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  const nn::Tensor pooled = nn::mean(x, {1, 2}, true);
  const nn::Tensor h =
      nn::relu(nn::conv2d(p, name + "/reduce", pooled, channels / reduction, 1, cfg));
  GateOutputs g;
  g.gate = nn::sigmoid(nn::conv2d(p, name + "/expand", h, channels, 1, cfg));
  g.output = x * g.gate;
  return g;
}

/// @brief Sequential channel and spatial attention tensor results.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned/state leaves remain owned by their stores; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::Cbam(p, state, "c", n::ones({2, 4, 4, 8}), 2, false, .9);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNet(), SqueezeExcitationModule(), Cbam(), Eca(), SelectiveKernel()
struct CbamOutputs {
  nn::Tensor channel_gate; ///< Logistic [N,1,1,C] gate from shared pooled MLP.
  nn::Tensor spatial_gate; ///< Logistic [N,H,W,1] gate after channel refinement.
  nn::Tensor output;       ///< Input multiplied by both gates with full gradients.
};
/// @brief Apply sequential shared-MLP channel and normalized spatial attention.
///
/// The SAME biased MLP C->floor(C/r)->C is evaluated on mean AND max
/// spatial pools, with ReLU internally; logits sum before sigmoid.
/// Refined features produce max-then-mean channel maps, concatenated into
/// a bias-free symmetric-padded 7x7 convolution, BN eps 1e-5, sigmoid.
/// MLP uses dense default initialization; spatial kernel uses He-uniform.
/// All gates remain differentiable, including max-pool selected entries.
///
/// @param p Mutable floating learned leaves.
/// @param state Mutable matching-dtype spatial BN moments.
/// @param name Prefix for shared /mlp1,/mlp2 and /spatial leaves.
/// @param x Nonempty floating NHWC input matching stores.
/// @param reduction Positive integer at most C.
/// @param training Use batch BN when true, saved moments when false.
/// @param momentum BN moving retention in [0,1).
/// @param update_stats True by default, permit eager training moment updates.
/// @return Channel/spatial gates and twice-rescaled NHWC output.
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::Cbam(p, state, "cb", n::ones({2, 5, 5, 8}), 2, false, .9);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline CbamOutputs Cbam(nn::ParameterStore &p, nn::ParameterStore &state,
                        const std::string &name, const nn::Tensor &x, std::int64_t reduction,
                        bool training, double momentum, bool update_stats = true) {
  if (x.dim() != 4 || x.numel() == 0 || reduction < 1 || reduction > x.size(3) ||
      !(momentum >= 0 && momentum < 1))
    throw std::invalid_argument("invalid CBAM dimensions/reduction/momentum");
  const std::int64_t channels = x.shape()[3];
  auto mlp = [&](const nn::Tensor &v) {
    const nn::Tensor h = nn::relu(nn::dense(p, name + "/mlp1", v, channels / reduction));
    return nn::dense(p, name + "/mlp2", h, channels);
  };
  CbamOutputs r;
  const nn::Tensor logits = mlp(nn::mean(x, {1, 2})) + mlp(nn::max(x, {1, 2}));
  r.channel_gate = nn::reshape(nn::sigmoid(logits), {-1, 1, 1, channels});
  const nn::Tensor refined = x * r.channel_gate;
  const nn::Tensor pooled =
      nn::concat({nn::max(refined, {3}, true), nn::mean(refined, {3}, true)}, 3);
  const nn::Tensor w = p.get(name + "/spatial/W", {7, 7, 2, 1}, nn::init::he_uniform());
  nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Explicit);
  o.explicit_paddings = {3, 3, 3, 3};
  nn::BatchNormOptions bn;
  bn.momentum = momentum;
  bn.eps = 1e-5;
  bn.update_stats = update_stats;
  r.spatial_gate = nn::sigmoid(
      nn::batch_norm(p, state, name + "/spatial/bn", nn::conv2d(pooled, w, o), training, bn));
  r.output = refined * r.spatial_gate;
  return r;
}

/// @brief Choose the reference adaptive odd ECA kernel size from channel width.
///
/// Architecture metadata only; O(1), no model tensors or fitting.
///
/// @param channels Positive channel count.
/// @return Odd t if t odd, else t+1, where t=floor(abs(log2(C)/2+.5)).
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// const int k = dm::nn::models::EcaKernel(2048); // 7
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline int EcaKernel(int channels) {
  if (channels < 1)
    throw std::invalid_argument("positive ECA channels required");
  const int t = int(std::abs(std::log2(double(channels)) / 2 + .5));
  return t % 2 ? t : t + 1;
}
/// @brief Apply bias-free local convolution along pooled channel coordinates.
///
/// Global spatial mean is treated as [N,1,C,1], convolved with zero
/// channel padding k/2 and a He-uniform filter. No reduction or BN.
/// Exactly k learned scalars, O(NHW*C+N*C*k) work, all gradients retained.
///
/// @param p Mutable floating store with name/W [1,k,1,1].
/// @param name Learned kernel prefix.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @param kernel Positive odd channel-neighbourhood size.
/// @return Logistic [N,1,1,C] gate and rescaled input.
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::Eca(p, "eca", n::ones({2, 5, 5, 8}), 3);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline GateOutputs Eca(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                       std::int64_t kernel) {
  if (x.dim() != 4 || x.numel() == 0 || kernel < 1 || kernel % 2 == 0)
    throw std::invalid_argument("ECA kernels are odd");
  const std::int64_t channels = x.shape()[3];
  // Channels as a length-C sequence: [N, 1, C, 1], convolved by a [1, k] kernel.
  const nn::Tensor pooled = nn::reshape(nn::mean(x, {1, 2}), {-1, 1, channels, 1});
  const nn::Tensor w = p.get(name + "/W", {1, kernel, 1, 1}, nn::init::he_uniform());
  nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Explicit);
  o.explicit_paddings = {0, 0, kernel / 2, kernel / 2};
  GateOutputs g;
  g.gate = nn::reshape(nn::sigmoid(nn::conv2d(pooled, w, o)), {-1, 1, 1, channels});
  g.output = x * g.gate;
  return g;
}

/// @brief Choose an optional attention gate before each residual shortcut sum.
///
/// Value metadata; no allocation. Selective kernels use the conv2 hook instead.
///
/// @par Example
/// @code
/// const auto mode = dm::nn::models::ChannelAttention::kEca;
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig
enum class ChannelAttention {
  kNone,              ///< No channel/spatial attention.
  kSqueezeExcitation, ///< Global-mean two-layer logistic channel gate.
  kCbam,              ///< Shared mean/max MLP followed by spatial attention.
  kEca                ///< Local channel convolution without dimensional reduction.
};

/// @brief Callable replacement for a bottleneck spatial stage.
///
/// Arguments are p,state,block prefix,NHWC input,width,ACTUAL conv2 stride,training.
/// Must return BN/ReLU-equivalent NHWC output with requested width/geometry.
/// std::function owns its closure; captured references must outlive calls and
/// any retained backward graph. Shared state access requires synchronization.
/// Empty function disables the hook; callbacks can throw to the caller.
///
/// @par Example
/// @code
/// dm::nn::models::Conv2Hook hook;
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig
using Conv2Hook =
    std::function<nn::Tensor(nn::ParameterStore &, nn::ParameterStore &, const std::string &,
                             const nn::Tensor &, std::int64_t, int, bool)>;
/// @brief Callable transformation applied after a bottleneck residual sum and ReLU.
///
/// Arguments are p,state,block prefix,NHWC output,zero-based stage,block,training.
/// Must preserve required width and stage geometry for subsequent blocks.
/// std::function owns its closure; captured references must outlive calls and
/// any retained backward graph. Shared state access requires synchronization.
/// Empty function disables the hook; callbacks can throw to the caller.
///
/// @par Example
/// @code
/// dm::nn::models::AfterBlockHook hook;
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig
using AfterBlockHook =
    std::function<nn::Tensor(nn::ParameterStore &, nn::ParameterStore &, const std::string &,
                             const nn::Tensor &, int, int, bool)>;

/// @brief Bottleneck residual architecture, attention choices and optional callbacks.
///
/// Configuration value metadata; no tensors or fitting. Callbacks are copied,
/// so their captured references must outlive each forward/backward that uses them.
///
/// @par Example
/// @code
/// dm::nn::models::AttentionResNetConfig cfg;
/// cfg.attention = dm::nn::models::ChannelAttention::kSqueezeExcitation;
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNet(), SqueezeExcitationModule(), Cbam(), Eca(), SelectiveKernel()
struct AttentionResNetConfig {
  int depth = 50;        ///< Bottleneck depth 50, 101 or 152.
  int classes = 1000;    ///< At least two classifier outputs.
  int width_divisor = 1; ///< Positive divisor at most 64; stage widths use integer division.
  int cardinality = 1;   ///< Positive group count for 3x3, one for ordinary ResNet.
  int base_width = 64;   ///< Positive base group width; ResNeXt uses four.
  int reduction = 16;    ///< Positive SE/CBAM divisor not exceeding branch output width.
  ChannelAttention attention =
      ChannelAttention::kNone; ///< Optional branch attention before shortcut sum.
  bool small_stem = false; ///< 3x3 stride-one stem without max pool instead of 7x7/2 and pool.
  bool stride_on_3x3 = true; ///< Stage stride belongs to conv2 when true, conv1 when false.
  bool adaptive_eca = false; ///< Choose per-stage ECA size from output channels.
  int eca_kernel = 3;        ///< Fixed positive odd ECA size when adaptive_eca is false.
  int deformable_from =
      0; ///< Zero disables deformable layers; 1..4 replaces this and later stages.
  bool modulated = false;   ///< Use logistic masks for deformable v2.
  double momentum = .9;     ///< BN moving-moment retention in [0,1).
  bool update_stats = true; ///< Permit eager BN state updates during training.
  Conv2Hook
      conv2; ///< Optional conv2 replacement; overrides default/deformable implementation.
  AfterBlockHook after_block; ///< Optional callback after every residual block.
};

/// @brief Intermediate maps and logits from a complete attention residual network.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned/state leaves remain owned by their stores; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::AttentionResNetConfig cfg;
/// cfg.width_divisor = 4;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::AttentionResNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNet(), SqueezeExcitationModule(), Cbam(), Eca(), SelectiveKernel()
struct AttentionResNetOutputs {
  nn::Tensor stem;                ///< Stem output after optional max pool.
  nn::Tensor features;            ///< Final block map after optional callback.
  nn::Tensor logits;              ///< [N,classes] scores before softmax.
  std::vector<nn::Tensor> blocks; ///< Every post-add ReLU map after optional callback.
};

/// @brief Build a complete bottleneck ResNet/ResNeXt with attention or spatial hooks.
///
/// Uses 3/4/6/3, 3/4/23/3 or 3/8/36/3 bottleneck stages. A symmetric
/// 7x7/2 stem and 3x3/2 pool can become a small 3x3/1 stem without pool.
/// Each unit has 1x1/BN/ReLU, spatial/BN/ReLU, linear 1x1/BN, optional
/// attention, shortcut sum/ReLU. Grouped width is floor(planes*base/64)*G;
/// ordinary width is planes. Stage strides belong to conv1 OR conv2, never
/// both; conv2 hooks receive that actual spatial stride. Projection shortcut
/// uses total stage stride. conv2 callback overrides deformable/default,
/// and after_block runs after each sum/ReLU. Hooks must preserve geometry.
/// He-uniform bias-free convolutions, BN eps 1e-5, global mean and biased
/// classifier. No preprocessing/loss/optimizer/fitting. Stores and callback
/// captures must outlive retained graphs; synchronize shared mutations.
///
/// @param p Mutable floating learned store, lazy named leaves.
/// @param state Mutable matching-dtype moving BN store.
/// @param image Nonempty floating NHWC input matching stores.
/// @param cfg Valid depth/scaling/groups/attention/BN metadata; default plain ResNet-50.
/// @param training True uses batch BN, optionally updates moments; false saved moments.
/// @return Stem, all post-block features, final map and [N,classes] logits.
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::AttentionResNetConfig cfg;
/// cfg.width_divisor = 4;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// cfg.attention = n::models::ChannelAttention::kEca;
/// const auto y = n::models::AttentionResNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline AttentionResNetOutputs AttentionResNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                              const nn::Tensor &image,
                                              const AttentionResNetConfig &cfg = {},
                                              bool training = false) {
  const auto counts = ResNetBlockCounts(cfg.depth);
  if (image.dim() != 4 || image.numel() == 0 || cfg.depth < 50 || cfg.classes < 2 ||
      cfg.width_divisor < 1 || cfg.width_divisor > 64 || cfg.cardinality < 1 ||
      cfg.base_width < 1 || cfg.reduction < 1 || !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      cfg.deformable_from < 0 || cfg.deformable_from > 4 || int(cfg.attention) < 0 ||
      int(cfg.attention) > 3 ||
      (cfg.attention == ChannelAttention::kEca && !cfg.adaptive_eca &&
       (cfg.eca_kernel < 1 || cfg.eca_kernel % 2 == 0)) ||
      ((cfg.attention == ChannelAttention::kSqueezeExcitation ||
        cfg.attention == ChannelAttention::kCbam) &&
       cfg.reduction > 4 * (64 / cfg.width_divisor)))
    throw std::invalid_argument("invalid attention ResNet configuration");
  for (int stage = 0; stage < 4; ++stage) {
    const int64_t planes = (64 << stage) / cfg.width_divisor,
                  group_width = planes * cfg.base_width / 64;
    if (cfg.cardinality > 1 &&
        (group_width < 1 ||
         group_width > std::numeric_limits<int64_t>::max() / cfg.cardinality))
      throw std::invalid_argument("invalid attention grouped width");
  }
  if (!cfg.conv2 && cfg.deformable_from > 0 && cfg.cardinality != 1)
    throw std::invalid_argument("deformable layers here are ungrouped");
  AttentionResNetOutputs out;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.epsilon = 1e-5;
  norm.momentum = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = 1e-5;
  bn.update_stats = cfg.update_stats;
  std::int64_t in = 64 / cfg.width_divisor;
  nn::Tensor h = ResNetConv(p, state, "stem", image, in, cfg.small_stem ? 3 : 7, norm,
                            cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t planes = (64 << stage) / cfg.width_divisor,
                       width = cfg.cardinality == 1
                                   ? planes
                                   : planes * cfg.base_width / 64 * cfg.cardinality,
                       outputs = 4 * planes;
    for (int block = 0; block < counts[stage]; ++block) {
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      const int stride = stage > 0 && block == 0 ? 2 : 1;
      nn::Tensor r = ResNetConv(p, state, name + "/conv1", h, width, 1, norm,
                                cfg.stride_on_3x3 ? 1 : stride);
      if (cfg.conv2) {
        r = cfg.conv2(p, state, name, r, width, cfg.stride_on_3x3 ? stride : 1, training);
      } else if (cfg.deformable_from > 0 && stage + 1 >= cfg.deformable_from) {
        if (cfg.cardinality != 1)
          throw std::invalid_argument("deformable layers here are ungrouped");
        r = DeformableConv2D(p, name + "/conv2", r, width, 3, cfg.stride_on_3x3 ? stride : 1,
                             cfg.modulated)
                .output;
        r = nn::relu(nn::batch_norm(p, state, name + "/conv2/bn", r, training, bn));
      } else if (cfg.cardinality == 1) {
        r = ResNetConv(p, state, name + "/conv2", r, width, 3, norm,
                       cfg.stride_on_3x3 ? stride : 1);
      } else {
        // One grouped convolution (tflow splits into per-group "<name>/conv2/group<g>/W" filters;
        // the parameter count and He-uniform fan-in are the same).
        const nn::Tensor w = p.get(name + "/conv2/W", {3, 3, width / cfg.cardinality, width},
                                   nn::init::he_uniform());
        nn::Conv2DOptions o =
            nn::conv_options(cfg.stride_on_3x3 ? stride : 1, nn::Padding::Explicit);
        o.explicit_paddings = {1, 1, 1, 1};
        o.groups = cfg.cardinality;
        r = nn::relu(
            nn::batch_norm(p, state, name + "/conv2/bn", nn::conv2d(r, w, o), training, bn));
      }
      r = ResNetConv(p, state, name + "/conv3", r, outputs, 1, norm, 1, false);
      switch (cfg.attention) {
      case ChannelAttention::kSqueezeExcitation:
        r = SqueezeExcitationModule(p, name + "/se", r, cfg.reduction).output;
        break;
      case ChannelAttention::kCbam:
        r = Cbam(p, state, name + "/cbam", r, cfg.reduction, training, cfg.momentum,
                 cfg.update_stats)
                .output;
        break;
      case ChannelAttention::kEca:
        r = Eca(p, name + "/eca", r,
                cfg.adaptive_eca ? EcaKernel(int(outputs)) : cfg.eca_kernel)
                .output;
        break;
      default:
        break;
      }
      const nn::Tensor shortcut =
          in == outputs && stride == 1
              ? h
              : ResNetConv(p, state, name + "/shortcut", h, outputs, 1, norm, stride, false);
      h = nn::relu(shortcut + r);
      in = outputs;
      if (cfg.after_block)
        h = cfg.after_block(p, state, name, h, stage, block, training);
      out.blocks.push_back(h);
    }
  }
  out.features = h;
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

// ---- Selective kernel (added by job 1.5b; tflow attention_cnn.h's SelectiveKernel) ----
// Selective kernel (https://arxiv.org/abs/1903.06586): two grouped 3x3
// branches (dilation one and two, i.e. 3x3 and 5x5 receptive fields) with
// BN/ReLU, their sum pooled and reduced to max(C/r, minimum) channels by a
// bias-free 1x1 with BN/ReLU, then per-branch bias-free 1x1 back to C and a
// softmax across branches weighting the branch outputs. The grouped branches
// are one grouped (dilated) nn::conv2d each, "<name>/branch<d>/W" (tflow:
// per-group filters "<name>/branch<d>/group<g>/W" with the same fans/count).
/// @brief Two spatial branch maps, normalized weights and their mixture.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned/state leaves remain owned by their stores; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::SelectiveKernel(p, state, "sk", n::ones({2, 5, 5, 8}), 8, 2, 1, 4, 4,
///                                           false, .9);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNet(), SqueezeExcitationModule(), Cbam(), Eca(), SelectiveKernel()
struct SelectiveKernelOutputs {
  std::vector<nn::Tensor> branches; ///< Dilation-one then dilation-two BN/ReLU maps.
  std::vector<nn::Tensor>
      weights;       ///< Two nonnegative [N,1,1,C] weights summing to one per channel.
  nn::Tensor output; ///< Differentiable weighted sum of the branch maps.
};
/// @brief Mix grouped dilation-one and dilation-two branches with per-channel softmax.
///
/// Both branches use grouped 3x3/BN/ReLU with symmetric dilation padding.
/// Their SUM is globally averaged and reduced to max(C/r,minimum) with
/// bias-free 1x1/BN/ReLU. Two bias-free 1x1 expansions produce logits,
/// stacked for softmax across the TWO BRANCHES per channel. No independent
/// sigmoid gates; weighted branch sum retains branch and selection derivatives.
/// Mathematically positive softmax weights can underflow to zero in floating arithmetic.
/// Dilation two reaches nine taps over 5x5, not a dense 5x5 filter. BN eps
/// 1e-5; He-uniform filters. No fitting or runtime measurement.
///
/// @param p Mutable floating learned store.
/// @param state Mutable matching-dtype BN moments.
/// @param name Prefix for branch1/branch2/fc/select0/select1 leaves.
/// @param x Nonempty floating NHWC input matching stores.
/// @param channels Positive branch output width divisible by groups.
/// @param groups Positive divisor of input/output widths.
/// @param stride Positive common branch spatial step.
/// @param reduction Positive bottleneck divisor.
/// @param minimum Positive lower bound on reduced width.
/// @param training Use batch BN when true.
/// @param momentum Moving BN retention in [0,1).
/// @param update_stats True by default, permit eager training moment updates.
/// @return Two branch maps, nonnegative weights summing to one, and weighted NHWC sum.
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::SelectiveKernel(p, state, "sk", n::ones({2, 5, 5, 8}), 8, 2, 1, 4, 4,
///                                           false, .9);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline SelectiveKernelOutputs SelectiveKernel(nn::ParameterStore &p, nn::ParameterStore &state,
                                              const std::string &name, const nn::Tensor &x,
                                              std::int64_t channels, std::int64_t groups,
                                              int stride, std::int64_t reduction,
                                              std::int64_t minimum, bool training,
                                              double momentum, bool update_stats = true) {
  if (x.dim() != 4 || x.numel() == 0 || channels < 1 || groups < 1 || x.size(3) % groups ||
      channels % groups || stride < 1 || reduction < 1 || minimum < 1 ||
      !(momentum >= 0 && momentum < 1))
    throw std::invalid_argument("invalid selective-kernel dimensions");
  nn::BatchNormOptions bn;
  bn.momentum = momentum;
  bn.eps = 1e-5;
  bn.update_stats = update_stats;
  auto norm = [&](const std::string &part, const nn::Tensor &h) {
    return nn::relu(nn::batch_norm(p, state, name + "/" + part + "/bn", h, training, bn));
  };
  const std::int64_t in = x.shape()[3];
  SelectiveKernelOutputs r;
  for (std::int64_t dilation : {1, 2}) {
    const auto part = "branch" + std::to_string(dilation);
    const nn::Tensor w =
        p.get(name + "/" + part + "/W", {3, 3, in / groups, channels}, nn::init::he_uniform());
    nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Explicit, dilation);
    o.explicit_paddings = {dilation, dilation, dilation, dilation};
    o.groups = groups;
    r.branches.push_back(norm(part, nn::conv2d(x, w, o)));
  }
  const std::int64_t reduced = std::max(channels / reduction, minimum);
  const nn::Tensor fused = r.branches[0] + r.branches[1];
  nn::ConvLayerOptions cfg;
  cfg.bias = false;
  cfg.init = nn::init::he_uniform();
  const nn::Tensor z =
      norm("fc", nn::conv2d(p, name + "/fc", nn::mean(fused, {1, 2}, true), reduced, 1, cfg));
  const nn::Tensor a = nn::conv2d(p, name + "/select0", z, channels, 1, cfg),
                   b = nn::conv2d(p, name + "/select1", z, channels, 1, cfg);
  const nn::Tensor soft = nn::softmax(nn::stack({a, b}, 4), 4); // [N, 1, 1, C, 2]
  const auto parts = nn::unstack(soft, 4);
  r.weights = {parts[0], parts[1]};
  r.output = r.branches[0] * parts[0] + r.branches[1] * parts[1];
  return r;
}
/// @brief Create a conv2 replacement closure that invokes selective-kernel filtering.
///
/// The closure forwards actual conv2 stride and training mode to
/// SelectiveKernel(). It creates no weights until called. Set hook
/// update_stats consistently with the outer config when freezing state.
///
/// @param cardinality Positive grouped branch count, must divide actual input/output width.
/// @param reduction Positive reduction divisor.
/// @param momentum BN retention in [0,1), default .9.
/// @param update_stats Permit eager training BN updates, default true.
/// @param minimum Positive reduced-width floor, default 32.
/// @return Owned callable capturing scalar options by value, using block prefix/sk.
/// @throws std::invalid_argument For invalid dimensions/settings, unsupported dtype, incompatible store entries or invalid normalization.
/// @par Example
/// @code
/// const auto hook = dm::nn::models::SelectiveKernelHook(32, 16);
/// @endcode
/// @par References
/// - Hu et al., https://arxiv.org/abs/1709.01507; Woo et al., https://arxiv.org/abs/1807.06521; Wang et al., https://arxiv.org/abs/1910.03151; Li et al., https://arxiv.org/abs/1903.06586
/// @see AttentionResNetConfig, AttentionResNet(), SelectiveKernel()
inline Conv2Hook SelectiveKernelHook(int cardinality, int reduction, double momentum = .9,
                                     bool update_stats = true, std::int64_t minimum = 32) {
  if (cardinality < 1 || reduction < 1 || minimum < 1 || !(momentum >= 0 && momentum < 1))
    throw std::invalid_argument("invalid selective-kernel hook");
  return [=](nn::ParameterStore &p, nn::ParameterStore &state, const std::string &name,
             const nn::Tensor &x, std::int64_t width, int stride, bool training) {
    return SelectiveKernel(p, state, name + "/sk", x, width, cardinality, stride, reduction,
                           minimum, training, momentum, update_stats)
        .output;
  };
}
} // namespace datamunge::nn::models
