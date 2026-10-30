#pragma once
/// @file mobilenet_models.hpp
/// @brief MobileNet-v1/v2/v3 tensor models, blocks and measured arithmetic metadata.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief One measured mobile layer output shape and convolution arithmetic budget.
///
/// Value metadata, not activation handles. A MAC is one multiply-accumulate pair.
/// Counts describe one input image, excluding bias, BN and activation work.
///
/// @par Example
/// @code
/// dm::nn::models::MobileLayer layer{"conv", {1, 8, 8, 16}, 432};
/// const auto macs = layer.Macs();
/// @endcode
/// @see MobileNetOutputs, ConvBN()
struct MobileLayer {
  std::string name; ///< Stable learned-layer prefix or pooled SE arithmetic label.
  nn::Shape shape;  ///< Measured output [N,H,W,C]; SE metadata uses [1,1,1,1].
  std::int64_t
      macs_per_position; ///< Kernel scalar products per output spatial position, all output channels.

  /// @brief Return convolution multiply-accumulates for one image at the measured shape.
  ///
  /// Requires caller-constructed metadata to have four shape entries and nonnegative
  /// counts. Model-generated metadata satisfies this. O(1), no model mutation.
  ///
  /// @return H*W*macs_per_position, excluding batch size.
  /// @throws std::invalid_argument For metadata without four axes or with negative sides/counts.
  /// @par Example
  /// @code
  /// const auto macs = dm::nn::models::MobileLayer{"c", {1, 8, 8, 16}, 432}.Macs(); // 27648
  /// @endcode
  /// @see MobileNetOutputs::Macs()
  std::int64_t Macs() const {
    if (shape.size() != 4 || shape[1] < 0 || shape[2] < 0 || macs_per_position < 0)
      throw std::invalid_argument("invalid mobile layer arithmetic metadata");
    return shape[1] * shape[2] * macs_per_position;
  }
};
/// @brief Borrowed parameter/state stores, layer log and BN mode for mobile forwards.
///
/// Referenced stores/vector must outlive the synchronous layer call. Copies retain
/// references; callers synchronize mutable stores, log and parameter generator.
/// No optimizer or fitting state is owned.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// @endcode
/// @see ConvBN(), ContextBatchNorm()
struct MobileContext {
  nn::ParameterStore &p;     ///< Borrowed floating learned leaves, lazily created/reused.
  nn::ParameterStore &state; ///< Borrowed same-dtype moving BN moments.
  std::vector<MobileLayer>
      &layers;              ///< Appended measured output metadata for every convolution/SE.
  bool training;            ///< Batch moments when true; saved moments in inference.
  double momentum = .9;     ///< Moving retention in [0,1).
  bool update_stats = true; ///< Permit moving updates in training mode.
};
/// @brief Normalize a mobile activation using borrowed mode and explicit epsilon.
///
/// Training uses batch population moments; eager moving variance updates use
/// unbiased moments. update_stats=false freezes state but still uses batch
/// moments in training. No fitting; O(numel(x)) work.
///
/// @param c Mutable borrowed stores and training/update/momentum settings.
/// @param name BN leaf/moving-moment prefix.
/// @param x Nonempty floating NHWC map matching store dtype.
/// @param epsilon Finite positive denominator offset; mobile releases use .001.
/// @return Same-shaped learned-scale/shift BN tensor with gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::ContextBatchNorm(c, "bn", n::ones({2, 5, 5, 4}), .001);
/// @endcode
/// @par References
/// - Howard et al. (2017), https://arxiv.org/abs/1704.04861
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline nn::Tensor ContextBatchNorm(MobileContext &c, const std::string &name,
                                   const nn::Tensor &x, double epsilon) {
  nn::BatchNormOptions bn;
  bn.momentum = c.momentum;
  bn.eps = epsilon;
  bn.update_stats = c.update_stats;
  return nn::batch_norm(c.p, c.state, name, x, c.training, bn);
}

/// @brief Choose dense connectivity or depthwise channelwise spatial filters.
///
/// Depthwise uses multiplier one and requires input/output channels to match.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::ConvBN(c, "d", n::ones({1, 5, 5, 4}), 4, 3, 1, n::Activation::Relu6,
///                                  n::models::MobileConv::kDepthwise);
/// @endcode
/// @par References
/// - Howard et al. (2017), https://arxiv.org/abs/1704.04861
/// @see ConvBN()
enum class MobileConv {
  kDense,    ///< Every output channel sees every input channel.
  kDepthwise ///< One spatial filter per input channel, with depth multiplier one.
};
/// @brief Apply a logged dense/depthwise SAME convolution, optional BN and activation.
///
/// Uses He-uniform fan-in; depthwise fan is kernel². Records kernel²*Cin*Cout
/// products per output position for dense or kernel²*Cin for depthwise.
/// No loss or fitting. BN is applied before activation, with learned gamma/beta.
///
/// @param c Borrowed stores and mutable layer log; appends one output metadata entry.
/// @param name Prefix for convolution W and optional b/bn leaves.
/// @param x Nonempty floating NHWC input matching store dtype.
/// @param out Positive output width; depthwise requires out==Cin.
/// @param kernel Positive square filter side.
/// @param stride Positive spatial stride.
/// @param act Activation after optional BN, including None for linear projection.
/// @param kind Dense or multiplier-one depthwise connectivity, default Dense.
/// @param normalize True uses BN eps .001 and no bias; false uses a biased filter.
/// @return Differentiable NHWC map with ceil spatial dimensions under SAME padding.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::ConvBN(c, "c", n::ones({1, 5, 5, 4}), 8, 3, 1, n::Activation::Relu6);
/// @endcode
/// @par References
/// - Howard et al. (2017), https://arxiv.org/abs/1704.04861
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline nn::Tensor ConvBN(MobileContext &c, const std::string &name, const nn::Tensor &x,
                         std::int64_t out, std::int64_t kernel, std::int64_t stride,
                         nn::Activation act, MobileConv kind = MobileConv::kDense,
                         bool normalize = true) {
  const std::int64_t in = x.dim() == 4 ? x.shape()[3] : 0;
  if (x.numel() == 0 || in < 1 || out < 1 || kernel < 1 || stride < 1 ||
      (kind != MobileConv::kDense && kind != MobileConv::kDepthwise) ||
      (kind == MobileConv::kDepthwise && out != in))
    throw std::invalid_argument("invalid mobile convolution");
  nn::ConvLayerOptions o;
  o.bias = !normalize;
  o.init = nn::init::he_uniform();
  o.stride = stride;
  o.padding = nn::Padding::Same;
  nn::Tensor h = kind == MobileConv::kDepthwise
                     ? nn::depthwise_conv2d(c.p, name, x, 1, kernel, o)
                     : nn::conv2d(c.p, name, x, out, kernel, o);
  if (normalize)
    h = ContextBatchNorm(c, name + "/bn", h, 1e-3);
  h = nn::activate(h, act);
  c.layers.push_back(
      {name, h.shape(), kernel * kernel * (kind == MobileConv::kDepthwise ? 1 : in) * out});
  return h;
}

/// @brief MobileNet-v1 dimensions, stem and normalization settings.
///
/// Value metadata; the forward function validates it before creating parameters.
/// Defaults use the full 1000-class ImageNet topology. No fitting or preprocessing.
///
/// @par Example
/// @code
/// dm::nn::models::MobileNetV1Config cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Howard et al. (2017), https://arxiv.org/abs/1704.04861
/// @see MobileNetV1()
struct MobileNetV1Config {
  double alpha = 1;   ///< Finite positive width multiplier; v1 truncates alpha*c, minimum one.
  int classes = 1000; ///< At least two output logits.
  bool small_stem = false; ///< First convolution stride one instead of two for small images.
  bool full_convolutions =
      false; ///< Replace every depthwise/pointwise pair by dense 3x3 (Table 4 baseline).
  double momentum = .9;     ///< BN moving-moment retention in [0,1).
  bool update_stats = true; ///< Allow moving updates during training.
};
/// @brief Mobile classifier features/logits and measured per-image arithmetic metadata.
///
/// Tensors retain storage/graphs; layer metadata is copied by value. Stores own
/// learned leaves and moving moments. MAC counts exclude BN, activations, pooling,
/// residual additions, SE channel scaling and bias operations; no timing is measured.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::MobileNetV1(p, state, n::ones({1, 32, 32, 3}));
/// const auto macs = y.Macs();
/// @endcode
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
struct MobileNetOutputs {
  nn::Tensor features;             ///< Last spatial map before global pooling/head dropout.
  nn::Tensor logits;               ///< Unnormalised [N,classes] classifier output.
  std::vector<MobileLayer> layers; ///< Ordered measured convolution and pooled-SE costs.
  std::int64_t classifier_macs =
      0; ///< Classifier matrix products, including v3 hidden dense head.

  /// @brief Sum every measured convolution/SE layer and classifier MACs for one image.
  ///
  /// O(number of recorded layers); uses actual output sides so resolution changes
  /// are counted rather than approximated by a scaling law. No model mutation.
  ///
  /// @return 64-bit per-image multiply-accumulate count, excluding batch size.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::ParameterStore p(17), state(17);
  /// const auto y = n::models::MobileNetV1(p, state, n::zeros({1, 32, 32, 3}));
  /// const auto count = y.Macs();
  /// @endcode
  /// @see MobileLayer::Macs()
  std::int64_t Macs() const {
    std::int64_t total = classifier_macs;
    for (const auto &l : layers)
      total += l.Macs();
    return total;
  }
};
/// @brief Build complete MobileNet-v1 tensor logits without fitting.
///
/// Thirteen depthwise-3x3/pointwise-1x1 pairs follow a 3x3 stem, all
/// BN/ReLU6. Widths truncate alpha*c with minimum one. Full-convolution
/// baseline replaces each pair by one dense 3x3. No dropout. Full ImageNet
/// learned/state counts 4231976/21888, with 568740352 MACs at 224 pixels.
/// All filters use He-uniform fan-in, BN eps .001 with learned gamma/beta.
/// small_stem changes only the first stride, retaining all remaining blocks.
/// No preprocessing, loss, optimizer or fitting is hidden. Training dropout
/// (v2/v3) draws from p.generator(); inference is deterministic after lazy
/// initialization. Work is logged per-image convolution/classifier MACs;
/// these exclude elementwise/pooling work and are not measured latency.
///
/// @param p Mutable floating learned store, lazily creates/reuses named leaves.
/// @param state Mutable BN moving moments matching p dtype.
/// @param image Nonempty floating NHWC images, positive channels.
/// @param cfg Valid dimensions, probabilities and BN settings, default full ImageNet topology.
/// @param training True uses batch moments and configured dropout, false saved moments.
/// @return Final spatial features, logits and measured arithmetic metadata with gradients.
/// @throws std::invalid_argument For invalid configuration, NHWC, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::MobileNetV1Config cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::MobileNetV1(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Howard et al. (2017), https://arxiv.org/abs/1704.04861
/// @see MobileNetV1Config, MobileNetOutputs
inline MobileNetOutputs MobileNetV1(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &image, const MobileNetV1Config &cfg = {},
                                    bool training = false) {
  if (image.dim() != 4 || image.numel() == 0 || !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      !std::isfinite(cfg.alpha) || cfg.alpha > double(std::numeric_limits<int>::max() / 1280))
    throw std::invalid_argument("invalid MobileNet input or normalization/width settings");
  if (!(cfg.alpha > 0) || cfg.classes < 2)
    throw std::invalid_argument("invalid MobileNet v1 configuration");
  MobileNetOutputs out;
  MobileContext c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  const auto width = [&](int channels) {
    return std::max(1, int(channels * cfg.alpha));
  };
  const int layers[13][2] = {{64, 1},  {128, 2},  {128, 1}, {256, 2}, {256, 1},
                             {512, 2}, {512, 1},  {512, 1}, {512, 1}, {512, 1},
                             {512, 1}, {1024, 2}, {1024, 1}};
  int in = width(32);
  nn::Tensor h =
      ConvBN(c, "conv0", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::Relu6);
  for (int i = 0; i < 13; ++i) {
    const int o = width(layers[i][0]), stride = layers[i][1];
    const auto name = "block" + std::to_string(i + 1);
    if (cfg.full_convolutions)
      h = ConvBN(c, name + "/conv", h, o, 3, stride, nn::Activation::Relu6);
    else {
      h = ConvBN(c, name + "/depthwise", h, in, 3, stride, nn::Activation::Relu6,
                 MobileConv::kDepthwise);
      h = ConvBN(c, name + "/pointwise", h, o, 1, 1, nn::Activation::Relu6);
    }
    in = o;
  }
  out.features = h;
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  out.classifier_macs = std::int64_t(in) * cfg.classes;
  return out;
}

// MobileNet v2 (https://arxiv.org/abs/1801.04381, torchvision/TF-slim
// releases): inverted residuals expand by t with a 1x1 (skipped when t=1),
// filter with a depthwise 3x3 (stride on the first repeat), and project with
// a linear 1x1 (BN, no activation), adding the input when shape is kept.
// Channels are alpha-scaled and rounded to multiples of eight (never down by
// more than 10%); the final 1x1 to 1280 channels only grows (alpha > 1). The
// head is global average pooling, dropout(0.2) in training and a classifier.
/// @brief Round a channel width to a positive divisor multiple with a ten-percent safeguard.
///
/// Rounds by int(value+divisor/2)/divisor, then enforces the .9 bound.
/// O(1), metadata only; no tensors, state or fitting.
///
/// @param value Finite nonnegative target width in the representable integer range.
/// @param divisor Positive channel alignment, default eight.
/// @return At least divisor, nearest aligned width, increased by divisor if below .9*value.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// const auto channels = dm::nn::models::MakeDivisible(9); // 16
/// @endcode
/// @par References
/// - Sandler et al. (2018), https://arxiv.org/abs/1801.04381
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline int MakeDivisible(double value, int divisor = 8) {
  if (!std::isfinite(value) || value < 0 || divisor < 1 ||
      value > double(std::numeric_limits<int>::max() - divisor))
    throw std::invalid_argument("invalid aligned channel width");
  int rounded = std::max(divisor, int(value + divisor / 2.) / divisor * divisor);
  return rounded < .9 * value ? rounded + divisor : rounded;
}
/// @brief Expanded, depthwise-filtered, linearly projected and residual-added tensors.
///
/// Handles retain storage and gradients. Expanded aliases input when expansion is one.
/// No activation follows projection or addition, preserving signed bottleneck values.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::InvertedResidual(c, "b", n::ones({1, 5, 5, 8}), 8, 6, 1);
/// @endcode
/// @par References
/// - Sandler et al. (2018), https://arxiv.org/abs/1801.04381
/// @see InvertedResidual()
struct InvertedResidualOutputs {
  nn::Tensor expanded;  ///< ReLU6 1x1 expansion or original input when t=1.
  nn::Tensor filtered;  ///< ReLU6 depthwise 3x3, possibly strided.
  nn::Tensor projected; ///< Linear 1x1 with BN, no activation.
  nn::Tensor output;    ///< Input+projected only at matching channels and stride one.
};
/// @brief Expand, depthwise-filter and linearly project an inverted residual unit.
///
/// Expansion/filtering use ReLU6; projection uses BN without activation.
/// Addition occurs only for stride one and matching widths, without a following
/// ReLU. Zero projection BN gives signed identity. No fitting. Dominant work
/// O(NHW*(Cin²*t+9*Cin*t+Cin*t*Cout)).
///
/// @param c Borrowed stores/log and BN mode.
/// @param name Prefix for optional expand, depthwise and project layers.
/// @param x Nonempty floating NHWC map matching store dtype.
/// @param out Positive linear bottleneck output width.
/// @param expansion Positive integer t; expansion is omitted when t=1.
/// @param stride One or two, applied only in depthwise 3x3.
/// @return Expanded/filtered/projected/output handles, all retaining gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::InvertedResidual(c, "b", n::ones({1, 5, 5, 8}), 8, 6, 1);
/// @endcode
/// @par References
/// - Sandler et al. (2018), https://arxiv.org/abs/1801.04381
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline InvertedResidualOutputs InvertedResidual(MobileContext &c, const std::string &name,
                                                const nn::Tensor &x, std::int64_t out,
                                                int expansion, int stride) {
  InvertedResidualOutputs r;
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || expansion < 1 ||
      (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid inverted residual dimensions");
  const std::int64_t in = x.shape()[3], hidden = in * expansion;
  r.expanded =
      expansion == 1 ? x : ConvBN(c, name + "/expand", x, hidden, 1, 1, nn::Activation::Relu6);
  r.filtered = ConvBN(c, name + "/depthwise", r.expanded, hidden, 3, stride,
                      nn::Activation::Relu6, MobileConv::kDepthwise);
  r.projected = ConvBN(c, name + "/project", r.filtered, out, 1, 1, nn::Activation::None);
  r.output = stride == 1 && in == out ? x + r.projected : r.projected;
  return r;
}
/// @brief MobileNet-v2 dimensions, stem and normalization settings.
///
/// Value metadata; the forward function validates it before creating parameters.
/// Defaults use the full 1000-class ImageNet topology. No fitting or preprocessing.
///
/// @par Example
/// @code
/// dm::nn::models::MobileNetV2Config cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Sandler et al. (2018), https://arxiv.org/abs/1801.04381
/// @see MobileNetV2()
struct MobileNetV2Config {
  double alpha = 1;   ///< Finite positive width multiplier, rounded with MakeDivisible(...,8).
  int classes = 1000; ///< At least two output logits.
  bool small_stem = false; ///< First convolution stride one instead of two.
  double keep_probability =
      .8;                   ///< Classifier input dropout retention in (0,1], training only.
  double momentum = .9;     ///< BN moving-moment retention in [0,1).
  bool update_stats = true; ///< Allow moving updates during training.
};
/// @brief Build complete MobileNet-v2 tensor logits without fitting.
///
/// Seventeen Table-2 inverted residuals use ReLU6 expansion/filtering and
/// linear bottlenecks. Widths align to eight with ten-percent safeguard;
/// final width 1280 grows only when alpha>1. Global mean, training dropout
/// and dense classifier. Full learned/state counts 3504872/34112 at alpha=1.
/// All filters use He-uniform fan-in, BN eps .001 with learned gamma/beta.
/// small_stem changes only the first stride, retaining all remaining blocks.
/// No preprocessing, loss, optimizer or fitting is hidden. Training dropout
/// (v2/v3) draws from p.generator(); inference is deterministic after lazy
/// initialization. Work is logged per-image convolution/classifier MACs;
/// these exclude elementwise/pooling work and are not measured latency.
///
/// @param p Mutable floating learned store, lazily creates/reuses named leaves.
/// @param state Mutable BN moving moments matching p dtype.
/// @param image Nonempty floating NHWC images, positive channels.
/// @param cfg Valid dimensions, probabilities and BN settings, default full ImageNet topology.
/// @param training True uses batch moments and configured dropout, false saved moments.
/// @return Final spatial features, logits and measured arithmetic metadata with gradients.
/// @throws std::invalid_argument For invalid configuration, NHWC, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::MobileNetV2Config cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::MobileNetV2(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Sandler et al. (2018), https://arxiv.org/abs/1801.04381
/// @see MobileNetV2Config, MobileNetOutputs
inline MobileNetOutputs MobileNetV2(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &image, const MobileNetV2Config &cfg = {},
                                    bool training = false) {
  if (image.dim() != 4 || image.numel() == 0 || !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      !std::isfinite(cfg.alpha) || cfg.alpha > double(std::numeric_limits<int>::max() / 1280))
    throw std::invalid_argument("invalid MobileNet input or normalization/width settings");
  if (!(cfg.alpha > 0) || cfg.classes < 2 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid MobileNet v2 configuration");
  MobileNetOutputs out;
  MobileContext c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  // (expansion t, channels c, repeats n, first stride s), Table 2.
  const int stages[7][4] = {{1, 16, 1, 1}, {6, 24, 2, 2},  {6, 32, 3, 2}, {6, 64, 4, 2},
                            {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1}};
  int in = MakeDivisible(32 * cfg.alpha);
  const int last = MakeDivisible(1280 * std::max(1., cfg.alpha));
  nn::Tensor h =
      ConvBN(c, "conv0", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::Relu6);
  int block = 0;
  for (const auto &stage : stages) {
    const int o = MakeDivisible(stage[1] * cfg.alpha);
    for (int i = 0; i < stage[2]; ++i, in = o)
      h = InvertedResidual(c, "block" + std::to_string(++block), h, o, stage[0],
                           i ? 1 : stage[3])
              .output;
  }
  out.features = ConvBN(c, "conv_last", h, last, 1, 1, nn::Activation::Relu6);
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training && cfg.keep_probability < 1)
    pooled = nn::dropout(pooled, 1 - cfg.keep_probability, true, p.generator());
  out.logits = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = std::int64_t(last) * cfg.classes;
  return out;
}

// MobileNet v3 (https://arxiv.org/abs/1905.02244, torchvision release):
// searched inverted residuals with 3x3 or 5x5 depthwise filters, ReLU or
// hard-swish, and optional squeeze-and-excitation on the depthwise output
// (global mean, 1x1 to MakeDivisible(expanded/4) with ReLU, 1x1 back with a
// hard sigmoid, channel scaling). The stem is 3x3/2 to 16 channels with
// hard-swish; the last stage is a 1x1 to six times the last width
// (BN, hard-swish), pooling, a biased dense layer to 1280 (Large) or 1024
// (Small) with hard-swish, dropout(0.2) in training and the classifier.
/// @brief Apply the MobileNet-v3 piecewise-linear sigmoid approximation.
///
/// Zero below/equal -3, one above/equal 3; slope 1/6 inside.
/// At hinges the core ReLU6 derivative convention applies. NaNs propagate.
/// No state or fitting; O(numel(x)) work/storage.
///
/// @param x Floating tensor of any shape; graph retained.
/// @return relu6(x+3)/6, same shape and dtype.
/// @par Example
/// @code
/// const auto y = dm::nn::models::HardSigmoid(dm::nn::tensor({-4., 0., 4.}));
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline nn::Tensor HardSigmoid(const nn::Tensor &x) {
  return nn::relu6(x + 3.0) / 6.0;
}
/// @brief Apply x times the MobileNet-v3 hard sigmoid gate.
///
/// Zero below/equal -3; identity above/equal 3; x*(x+3)/6 inside.
/// Interior derivative x/3+.5 may be negative. Hinge derivatives follow
/// core ReLU6; no state or fitting. O(numel(x)) work/storage.
///
/// @param x Floating tensor, any shape, graph retained.
/// @return x*relu6(x+3)/6, preserving shape and dtype.
/// @par Example
/// @code
/// const auto y = dm::nn::models::HardSwish(dm::nn::tensor({-4., 0., 4.}));
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline nn::Tensor HardSwish(const nn::Tensor &x) {
  return x * HardSigmoid(x);
}
/// @brief Hard-sigmoid channel gates and the scaled mobile feature map.
///
/// Handles retain storage/gradient graphs; gates are shared spatially within each image.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::SqueezeExcite(p, "se", n::ones({2, 5, 5, 8}), 8);
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see SqueezeExcite()
struct SqueezeExcitation {
  nn::Tensor gate; ///< Floating [N,1,1,C] hard-sigmoid gates in [0,1].
  nn::Tensor
      output; ///< NHWC input multiplied by its channel gates, all gradient paths retained.
};
/// @brief Gate NHWC channels from global pooled features using a hard sigmoid.
///
/// Global mean keeps spatial singleton axes; reduction/ReLU then expansion/
/// hard-sigmoid. He-uniform filters, zero biases, no BN. Gradients include
/// input dependence of the gates. No fitting. Work O(NHWC+N*C*squeeze).
///
/// @param p Mutable floating learned store, lazily creates/reuses two biased 1x1 filters.
/// @param name Prefix for reduce/W|b and expand/W|b.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @param squeeze Positive hidden channel count, normally MakeDivisible(expanded/4).
/// @return [N,1,1,C] gates and input-scaled NHWC output with complete derivatives.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::SqueezeExcite(p, "se", n::ones({2, 5, 5, 8}), 8);
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline SqueezeExcitation SqueezeExcite(nn::ParameterStore &p, const std::string &name,
                                       const nn::Tensor &x, std::int64_t squeeze) {
  if (x.dim() != 4 || x.numel() == 0 || squeeze < 1)
    throw std::invalid_argument("invalid squeeze/excitation dimensions");
  const std::int64_t channels = x.shape()[3];
  nn::ConvLayerOptions o;
  o.init = nn::init::he_uniform();
  const nn::Tensor pooled = nn::mean(x, {1, 2}, true);
  const nn::Tensor h = nn::relu(nn::conv2d(p, name + "/reduce", pooled, squeeze, 1, o));
  SqueezeExcitation se;
  se.gate = HardSigmoid(nn::conv2d(p, name + "/expand", h, channels, 1, o));
  se.output = x * se.gate;
  return se;
}
/// @brief One original MobileNet-v3 searched bottleneck specification.
///
/// Architecture metadata without allocation or fitting. Channel values are the
/// released Large/Small dimensions; this configuration has no width multiplier.
///
/// @par Example
/// @code
/// const auto blocks = dm::nn::models::MobileNetV3Blocks(true);
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV3Blocks(), MobileNetV3()
struct MobileNetV3Block {
  int in;          ///< Input bottleneck width, matching the preceding output.
  int kernel;      ///< Depthwise square filter width, three or five.
  int expanded;    ///< Internal expansion width before depthwise filtering.
  int out;         ///< Linear bottleneck output width.
  bool se;         ///< Apply hard-sigmoid squeeze/excitation after depthwise activation.
  bool hard_swish; ///< Use hard-swish instead of ReLU in this block.
  int stride;      ///< Depthwise stride, one or two.
};
/// @brief Return the original searched MobileNet-v3 Large or Small bottleneck table.
///
/// Contains exact channel/kernel/SE/activation/stride choices, before the
/// final efficient head. Architecture metadata only, no learned allocation.
///
/// @param large True selects Large, false Small.
/// @return Value vector with 15 Large or 11 Small block specifications.
/// @par Example
/// @code
/// const auto blocks = dm::nn::models::MobileNetV3Blocks(false); // 11 blocks
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV1(), MobileNetV2(), MobileNetV3()
inline std::vector<MobileNetV3Block> MobileNetV3Blocks(bool large) {
  if (large)
    return {{16, 3, 16, 16, false, false, 1},  {16, 3, 64, 24, false, false, 2},
            {24, 3, 72, 24, false, false, 1},  {24, 5, 72, 40, true, false, 2},
            {40, 5, 120, 40, true, false, 1},  {40, 5, 120, 40, true, false, 1},
            {40, 3, 240, 80, false, true, 2},  {80, 3, 200, 80, false, true, 1},
            {80, 3, 184, 80, false, true, 1},  {80, 3, 184, 80, false, true, 1},
            {80, 3, 480, 112, true, true, 1},  {112, 3, 672, 112, true, true, 1},
            {112, 5, 672, 160, true, true, 2}, {160, 5, 960, 160, true, true, 1},
            {160, 5, 960, 160, true, true, 1}};
  return {{16, 3, 16, 16, true, false, 2},  {16, 3, 72, 24, false, false, 2},
          {24, 3, 88, 24, false, false, 1}, {24, 5, 96, 40, true, true, 2},
          {40, 5, 240, 40, true, true, 1},  {40, 5, 240, 40, true, true, 1},
          {40, 5, 120, 48, true, true, 1},  {48, 5, 144, 48, true, true, 1},
          {48, 5, 288, 96, true, true, 2},  {96, 5, 576, 96, true, true, 1},
          {96, 5, 576, 96, true, true, 1}};
}
/// @brief MobileNet-v3 dimensions, stem and normalization settings.
///
/// Value metadata; the forward function validates it before creating parameters.
/// Defaults use the full 1000-class ImageNet topology. No fitting or preprocessing.
///
/// @par Example
/// @code
/// dm::nn::models::MobileNetV3Config cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV3()
struct MobileNetV3Config {
  bool large = true;            ///< Released Large (15 blocks) or Small (11 blocks).
  bool small_stem = false;      ///< First convolution stride one instead of two.
  int classes = 1000;           ///< At least two output logits.
  double keep_probability = .8; ///< Head dropout retention in (0,1], training only.
  double momentum = .9;         ///< BN moving-moment retention in [0,1).
  bool update_stats = true;     ///< Allow moving updates during training.
};
/// @brief Build complete MobileNet-v3 tensor logits without fitting.
///
/// Large/Small use the exact 15/11 searched blocks, 3x3/5x5 depthwise
/// filters, ReLU/hard-swish and optional hard-sigmoid SE. Final spatial 1x1
/// width is six times the last bottleneck. After pooling a biased dense
/// hard-swish head has 1280/1024 units, training dropout and classifier.
/// Full Large/Small learned counts 5483032/2542856.
/// All filters use He-uniform fan-in, BN eps .001 with learned gamma/beta.
/// small_stem changes only the first stride, retaining all remaining blocks.
/// No preprocessing, loss, optimizer or fitting is hidden. Training dropout
/// (v2/v3) draws from p.generator(); inference is deterministic after lazy
/// initialization. Work is logged per-image convolution/classifier MACs;
/// these exclude elementwise/pooling work and are not measured latency.
///
/// @param p Mutable floating learned store, lazily creates/reuses named leaves.
/// @param state Mutable BN moving moments matching p dtype.
/// @param image Nonempty floating NHWC images, positive channels.
/// @param cfg Valid dimensions, probabilities and BN settings, default full ImageNet topology.
/// @param training True uses batch moments and configured dropout, false saved moments.
/// @return Final spatial features, logits and measured arithmetic metadata with gradients.
/// @throws std::invalid_argument For invalid configuration, NHWC, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::MobileNetV3Config cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::MobileNetV3(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Howard et al. (2019), https://arxiv.org/abs/1905.02244
/// @see MobileNetV3Config, MobileNetOutputs
inline MobileNetOutputs MobileNetV3(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &image, const MobileNetV3Config &cfg = {},
                                    bool training = false) {
  if (image.dim() != 4 || image.numel() == 0 || !(cfg.momentum >= 0 && cfg.momentum < 1))
    throw std::invalid_argument("invalid MobileNet input or normalization/width settings");
  if (cfg.classes < 2 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid MobileNet v3 configuration");
  MobileNetOutputs out;
  MobileContext c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  auto act = [](const nn::Tensor &x, bool hard_swish) {
    return hard_swish ? HardSwish(x) : nn::relu(x);
  };
  nn::Tensor h = HardSwish(
      ConvBN(c, "conv0", image, 16, 3, cfg.small_stem ? 1 : 2, nn::Activation::None));
  const auto blocks = MobileNetV3Blocks(cfg.large);
  int index = 0;
  for (const auto &b : blocks) {
    const auto name = "block" + std::to_string(++index);
    nn::Tensor x = h;
    if (b.expanded != b.in)
      h = act(ConvBN(c, name + "/expand", h, b.expanded, 1, 1, nn::Activation::None),
              b.hard_swish);
    h = act(ConvBN(c, name + "/depthwise", h, b.expanded, b.kernel, b.stride,
                   nn::Activation::None, MobileConv::kDepthwise),
            b.hard_swish);
    if (b.se) {
      const int squeeze = MakeDivisible(b.expanded / 4);
      h = SqueezeExcite(p, name + "/se", h, squeeze).output;
      // Two 1x1 layers on the pooled vector (the channel scaling is not counted).
      c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * std::int64_t(b.expanded) * squeeze});
    }
    h = ConvBN(c, name + "/project", h, b.out, 1, 1, nn::Activation::None);
    if (b.stride == 1 && b.in == b.out)
      h = x + h;
  }
  const int width = blocks.back().out, last = 6 * width, head = cfg.large ? 1280 : 1024;
  out.features = HardSwish(ConvBN(c, "conv_last", h, last, 1, 1, nn::Activation::None));
  nn::Tensor v = HardSwish(nn::dense(p, "head", nn::global_avg_pool2d(out.features), head));
  if (training && cfg.keep_probability < 1)
    v = nn::dropout(v, 1 - cfg.keep_probability, true, p.generator());
  out.logits = nn::dense(p, "output", v, cfg.classes);
  out.classifier_macs = std::int64_t(last) * head + std::int64_t(head) * cfg.classes;
  return out;
}
} // namespace datamunge::nn::models
