#pragma once
/// @file reversible_residual_models.hpp
/// @brief RevNet coupling functions and complete tensor classifiers with reversible backward.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Stage layout and preactivation options for released reversible residual networks.
///
/// Default CIFAR-110 has 9 units per stage and basic coupling functions.
/// Configuration is value metadata without model allocation or fitting.
///
/// @par Example
/// @code
/// const auto cfg = dm::nn::models::RevNetConfig::Cifar(38);
/// const auto widths = cfg.Widths();
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet()
struct RevNetConfig {
  std::vector<int> units = {9, 9, 9}; ///< Positive coupling-unit counts for each stage.
  std::vector<int> filters = {32, 32, 64, 128}; ///< Stem width then stage inner/basic widths.
  std::vector<int> strides = {1, 2, 2}; ///< Stage-entry spatial strides, each one or two.
  bool bottleneck = false; ///< Expand stage filter widths fourfold for bottleneck couplings.
  bool imagenet = false;   ///< Use 7x7/2 stem and initial SAME max-pool.
  int classes = 10;        ///< At least two output logits.
  double momentum = .9;    ///< Moving BN retention in [0,1).

  /// @brief Construct a released CIFAR RevNet-38, 110 or 164 preset.
  ///
  /// Depth 38 uses 3/3/3 units and stage widths 32/64/112. Depths 110/164
  /// use 9/9/9 units and filter widths 32/64/128, expanded for 164.
  ///
  /// @param depth Exactly 38, 110 or 164; 164 selects bottleneck functions.
  /// @param classes Output class count, default ten; Widths() validates at least two.
  /// @return Independent preset metadata; no parameters allocated.
  /// @throws std::invalid_argument For unsupported depth.
  /// @par Example
  /// @code
  /// const auto cfg = dm::nn::models::RevNetConfig::Cifar(38);
  /// @endcode
  /// @par References
  /// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
  /// @see ImageNet(), Widths()
  static RevNetConfig Cifar(int depth, int classes = 10) {
    RevNetConfig c;
    c.classes = classes;
    if (depth == 38) {
      c.units = {3, 3, 3};
      c.filters = {32, 32, 64, 112};
    } else if (depth == 164) {
      c.bottleneck = true;
    } else if (depth != 110) {
      throw std::invalid_argument("CIFAR RevNet depths are 38/110/164");
    }
    return c;
  }
  /// @brief Construct a released ImageNet RevNet-56 or 104 bottleneck preset.
  ///
  /// Unit counts are 2/2/3/2 or 2/2/11/2. Inner stage widths
  /// 128/256/512/832 expand fourfold; entry strides are 1/2/2/2.
  ///
  /// @param depth Exactly 56 or 104.
  /// @return Metadata with 1000 classes, four stages and a 128-channel stem.
  /// @throws std::invalid_argument For unsupported depth.
  /// @par Example
  /// @code
  /// const auto cfg = dm::nn::models::RevNetConfig::ImageNet(56);
  /// @endcode
  /// @par References
  /// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
  /// @see Cifar(), Widths()
  static RevNetConfig ImageNet(int depth) {
    RevNetConfig c;
    c.imagenet = c.bottleneck = true;
    c.classes = 1000;
    c.filters = {128, 128, 256, 512, 832};
    c.strides = {1, 2, 2, 2};
    if (depth == 56)
      c.units = {2, 2, 3, 2};
    else if (depth == 104)
      c.units = {2, 2, 11, 2};
    else
      throw std::invalid_argument("ImageNet RevNet depths are 56/104");
    return c;
  }
  // Channels entering stage 1, then leaving each stage.
  /// @brief Validate the full configuration and return stem/expanded stage widths.
  ///
  /// Requires at least one stage, matching vectors, positive unit counts, stride
  /// one/two, even nondecreasing widths and valid classes/momentum. Basic stage
  /// increases must be multiples of four for symmetric padding of both halves.
  /// Work/storage O(stages); does not mutate configuration or learned state.
  ///
  /// @return Copy of filters with every post-stem entry multiplied by four for bottlenecks.
  /// @throws std::invalid_argument For invalid layout, widths, classes or momentum.
  /// @par Example
  /// @code
  /// const auto w = dm::nn::models::RevNetConfig::Cifar(38).Widths(); // {32,32,64,112}
  /// @endcode
  /// @see RevNetUnits()
  std::vector<int> Widths() const {
    if (units.empty() || filters.size() != units.size() + 1 ||
        strides.size() != units.size() || classes < 2 || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid RevNet configuration");
    std::vector<int> w = filters;
    if (bottleneck)
      for (std::size_t i = 1; i < w.size(); ++i)
        w[i] *= 4;
    for (std::size_t i = 0; i < w.size(); ++i)
      if (w[i] < 2 || w[i] % 2 ||
          (i && (w[i] < w[i - 1] || (!bottleneck && (w[i] - w[i - 1]) % 4))))
        throw std::invalid_argument("invalid RevNet widths");
    for (std::size_t i = 0; i < units.size(); ++i)
      if (units[i] < 1 || (strides[i] != 1 && strides[i] != 2))
        throw std::invalid_argument("invalid RevNet stage");
    return w;
  }
};

// Normalization context: training mode uses batch statistics; moving
// averages are updated only when `update` is set (and never while a unit is
// recomputed during the backward pass).
/// @brief Borrowed stores and BN settings captured by reversible coupling functions.
///
/// p/state references must outlive all returned forward and recomputation graphs.
/// Copies retain references, not independent stores. Callers must prevent concurrent
/// mutation and parameter changes between forward and backward. No fitting occurs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// @endcode
/// @see RevCoupling(), RevNormRelu()
struct RevContext {
  nn::ParameterStore &p; ///< Borrowed mutable learned leaves, lazily created by forward calls.
  nn::ParameterStore &state; ///< Borrowed moving BN moments, same dtype as p.
  bool training;             ///< Use batch moments when true, saved moments otherwise.
  double momentum;           ///< BN moving retention in [0,1).
  bool
      bottleneck; ///< Three-filter bottleneck functions instead of basic two-filter functions.
  bool update =
      true; ///< Permit moving updates in forward, suppressed automatically during recomputation.
};
/// @brief Two equal-shaped NHWC channel halves for additive coupling.
///
/// Tensor handles retain storage and autograd graphs; no data is copied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// n::models::RevPair x{n::ones({2, 5, 5, 4}), n::ones({2, 5, 5, 4})};
/// @endcode
/// @see RevBlockForward(), RevBlockInverse()
struct RevPair {
  nn::Tensor first;  ///< First half, matching second in shape and dtype.
  nn::Tensor second; ///< Second half; each last dimension is half the full unit width.
};

/// @brief Apply a bias-free He-uniform SAME convolution for a reversible function.
///
/// No normalization, activation or fitting. SAME padding follows core convolution
/// semantics, including asymmetric padding at strided boundaries. Work
/// O(NHoWo*kernel²*Cin*out). He-uniform replaces released truncated-normal fan-out.
///
/// @param p Mutable learned store, lazily creates/reuses name/W.
/// @param name Filter prefix.
/// @param x Nonempty floating NHWC input, matching p dtype.
/// @param out Positive output channel width.
/// @param kernel Positive square spatial filter width.
/// @param stride Positive spatial step, default one.
/// @return Differentiable NHWC convolution, sides ceil(input/stride).
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// const auto y = n::models::RevConv(p, "c", n::ones({1, 5, 5, 4}), 4, 3);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline nn::Tensor RevConv(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                          std::int64_t out, std::int64_t kernel, std::int64_t stride = 1) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || kernel < 1 || stride < 1)
    throw std::invalid_argument("invalid RevConv dimensions");
  nn::ConvLayerOptions o;
  o.bias = false;
  o.init = nn::init::he_uniform();
  o.stride = stride;
  o.padding = nn::Padding::Same;
  return nn::conv2d(p, name, x, out, kernel, o);
}
/// @brief Apply learned-scale BN with epsilon .001, then ReLU.
///
/// Training uses batch moments. Moving averages update only if c.update and
/// not nn::in_recompute(), preventing checkpoint/reversible backward updates
/// from applying twice. Inference reads saved moments. O(numel(x)) work.
///
/// @param c Borrowed model stores and training/update settings.
/// @param name BN leaf and moving-moment prefix.
/// @param x Floating NHWC activation matching the context dtype.
/// @return BN/ReLU tensor retaining gradients, same shape as x.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// const auto y = n::models::RevNormRelu(c, "bn", n::ones({2, 5, 5, 4}));
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline nn::Tensor RevNormRelu(const RevContext &c, const std::string &name,
                              const nn::Tensor &x) {
  nn::BatchNormOptions o;
  o.momentum = c.momentum;
  o.eps = 1e-3;
  o.update_stats = c.update && !nn::in_recompute();
  return nn::relu(nn::batch_norm(c.p, c.state, name, x, c.training, o));
}
// F or G: a pre-activation residual function from `in` to `out` channels.
/// @brief Compute one F or G preactivation residual function.
///
/// Basic uses preactivation before two 3x3 filters. Bottleneck uses 1x1/3x3/1x1
/// with inner width out/4. All convolutions are bias-free SAME. No fitting;
/// work/storage sums filters. Do not change stores before graph backward.
///
/// @param c Borrowed stores and basic/bottleneck selection.
/// @param name Prefix for bn1..bn2/3 and conv1..conv2/3.
/// @param x Nonempty floating NHWC input, handle copied without mutating storage.
/// @param out Positive output width; divisible by four for bottlenecks.
/// @param stride One or two, applied in the first convolution.
/// @param no_activation Skip the first BN/ReLU only for the first F after the stem.
/// @return Linear final convolution output without a post-branch activation.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// const auto y = n::models::RevResidual(c, "f", n::ones({2, 5, 5, 4}), 4, 1, false);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline nn::Tensor RevResidual(const RevContext &c, const std::string &name, nn::Tensor x,
                              int out, int stride, bool no_activation) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || (c.bottleneck && out % 4) ||
      (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid RevResidual dimensions");
  if (!no_activation)
    x = RevNormRelu(c, name + "/bn1", x);
  if (c.bottleneck) {
    x = RevConv(c.p, name + "/conv1", x, out / 4, 1, stride);
    x = RevConv(c.p, name + "/conv2", RevNormRelu(c, name + "/bn2", x), out / 4, 3);
    return RevConv(c.p, name + "/conv3", RevNormRelu(c, name + "/bn3", x), out, 1);
  }
  x = RevConv(c.p, name + "/conv1", x, out, 3, stride);
  return RevConv(c.p, name + "/conv2", RevNormRelu(c, name + "/bn2", x), out, 3);
}
/// @brief Apply an identity, symmetric zero-padded pooled path or bottleneck projection.
///
/// Basic shortcuts average-pool when strided then append half the extra zero
/// channels on each side. Bottleneck changes use a raw learned 1x1 SAME filter
/// without BN. Matching width/stride returns x. No fitting.
///
/// @param c Borrowed context; bottleneck selects learned projections.
/// @param name Prefix for optional /projection/W.
/// @param x Nonempty NHWC half-map, last dimension exactly in.
/// @param in Positive input half-width.
/// @param out Nondecreasing output half-width; basic difference must be even.
/// @param stride One or two; basic stride-two requires even spatial sides.
/// @return Identity/pooled-padded or projected NHWC half-map with gradients.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// const auto y = n::models::RevShortcut(c, "skip", n::ones({1, 4, 4, 4}), 4, 6, 2);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline nn::Tensor RevShortcut(const RevContext &c, const std::string &name, nn::Tensor x,
                              int in, int out, int stride) {
  if (x.dim() != 4 || x.numel() == 0 || in < 1 || x.size(3) != in || out < in ||
      (stride != 1 && stride != 2) ||
      (!c.bottleneck && ((out - in) % 2 || (stride == 2 && (x.size(1) % 2 || x.size(2) % 2)))))
    throw std::invalid_argument("invalid reversible shortcut geometry");
  if (c.bottleneck)
    return stride > 1 || in != out ? RevConv(c.p, name + "/projection", x, out, 1, stride) : x;
  if (stride > 1)
    x = nn::avg_pool2d(x, stride, stride, nn::Padding::Valid);
  if (in == out)
    return x;
  const std::int64_t pad = (out - in) / 2;
  return nn::pad(x, {{0, 0}, {0, 0}, {0, 0}, {pad, pad}});
}
/// @brief Apply additive coupling y1=S(x1)+F(x2), y2=S(x2)+G(y1).
///
/// Only equal-width stride-one units with no_activation=false can use
/// RevBlockInverse. Stage entries change geometry or first-F behavior and their
/// inputs must be stored. There is no post-addition ReLU or optimizer.
///
/// @param c Borrowed stores, normalization and residual-function settings.
/// @param name Unit prefix with /f, /g, /x1 and /x2 branches.
/// @param x Equal-shaped nonempty NHWC halves of total width in.
/// @param in Positive even full input width.
/// @param out Even nondecreasing full output width; bottleneck output divisible by eight.
/// @param stride One or two; basic strided sides must be even.
/// @param no_activation Skip first F BN/ReLU, only for the first model unit.
/// @return Two differentiable coupled output halves of full width out.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// n::models::RevPair x{n::ones({2, 5, 5, 4}), n::ones({2, 5, 5, 4})};
/// const auto y = n::models::RevBlockForward(c, "u", x, 8, 8, 1, false);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline RevPair RevBlockForward(const RevContext &c, const std::string &name, const RevPair &x,
                               int in, int out, int stride, bool no_activation) {
  if (in < 2 || in % 2 || out < in || out % 2 || (c.bottleneck && out % 8) ||
      (!c.bottleneck && (out - in) % 4) || x.first.dim() != 4 || x.first.numel() == 0 ||
      x.first.shape() != x.second.shape() || x.first.size(3) != in / 2)
    throw std::invalid_argument("invalid reversible pair or widths");
  const int hi = in / 2, ho = out / 2;
  nn::Tensor y1 = RevResidual(c, name + "/f", x.second, ho, stride, no_activation) +
                  RevShortcut(c, name + "/x1", x.first, hi, ho, stride);
  nn::Tensor y2 = RevResidual(c, name + "/g", y1, ho, 1, false) +
                  RevShortcut(c, name + "/x2", x.second, hi, ho, stride);
  return {y1, y2};
}
// Inverse of an equal-width, stride-one unit.
/// @brief Reconstruct the inputs of an equal-width stride-one coupling unit.
///
/// Never use for stage entries, width/stride changes or skipped first F activation.
/// Pass identical parameter values and BN mode; inference also needs unchanged
/// moving moments. In training use update=false to avoid extra moment updates.
/// Finite precision causes reconstruction drift; this is not bitwise inversion.
/// No automatic gradient checkpointing is performed by this helper.
///
/// @param c Same stores and BN mode as the original forward; c.update should be false.
/// @param name Prefix of an existing unit, first-F preactivation enabled.
/// @param y Equal-shaped NHWC output halves.
/// @param channels Positive even full width, divisible by eight for bottlenecks.
/// @return Reconstructed x1,x2 via x2=y2-G(y1), x1=y1-F(x2), graph retained.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// n::models::RevPair x{n::ones({2, 5, 5, 4}), n::ones({2, 5, 5, 4})};
/// const auto y = n::models::RevBlockForward(c, "u", x, 8, 8, 1, false);
/// const auto recovered = n::models::RevBlockInverse(c, "u", y, 8);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline RevPair RevBlockInverse(const RevContext &c, const std::string &name, const RevPair &y,
                               int channels) {
  if (channels < 2 || channels % 2 || (c.bottleneck && channels % 8) || y.first.dim() != 4 ||
      y.first.numel() == 0 || y.first.shape() != y.second.shape() ||
      y.first.size(3) != channels / 2)
    throw std::invalid_argument("invalid inverse pair or width");
  const int h = channels / 2;
  nn::Tensor x2 = y.second - RevResidual(c, name + "/g", y.first, h, 1, false);
  nn::Tensor x1 = y.first - RevResidual(c, name + "/f", x2, h, 1, false);
  return {x1, x2};
}
// The coupling functions of an equal-width unit, for nn::reversible_sequence.
// `c` is copied: the backward pass calls F and G after this returns.
/// @brief Capture equal-width F and G callbacks for nn::reversible_sequence.
///
/// The callbacks may run after this call returns. Stores must outlive all graph
/// uses and weights may not change before backward. Moving updates are
/// suppressed during recomputation. No dropout or random operations occur.
///
/// @param c Context copied by value; its referenced stores remain borrowed.
/// @param name Prefix of an existing equal-width unit.
/// @param channels Positive even full width, divisible by eight for bottlenecks.
/// @return Two deterministic residual callbacks whose first BN/ReLU is enabled.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// n::models::RevPair x{n::ones({2, 5, 5, 4}), n::ones({2, 5, 5, 4})};
/// const auto coupling = n::models::RevCoupling(c, "u", 8);
/// const auto y = n::reversible_sequence({coupling}, x.first, x.second);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline nn::ReversibleBlock RevCoupling(const RevContext &c, const std::string &name,
                                       int channels) {
  if (channels < 2 || channels % 2 || (c.bottleneck && channels % 8))
    throw std::invalid_argument("invalid reversible coupling width");
  const int h = channels / 2;
  return {[c, name, h](const nn::Tensor &x2) {
            return RevResidual(c, name + "/f", x2, h, 1, false);
          },
          [c, name, h](const nn::Tensor &y1) {
            return RevResidual(c, name + "/g", y1, h, 1, false);
          }};
}
/// @brief Concatenate halves, normalize/ReLU, global-average pool and classify.
///
/// Uses final/bn and output leaf names. Dense uses Glorot with bias. All
/// returned handles own storage references; no fitting. Work O(NHWC+NC*classes).
///
/// @param c Borrowed learned/state stores and normalization settings.
/// @param x Matching NHWC halves after the last unit.
/// @param classes Output width, at least two.
/// @param features Optional borrowed output pointer, assigned the pre-pool BN/ReLU handle.
/// @return Unnormalised [N,classes] dense logits with gradients.
/// @throws std::invalid_argument For invalid input geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RevContext c{p, state, false, .9, false, false};
/// n::models::RevPair x{n::ones({2, 5, 5, 4}), n::ones({2, 5, 5, 4})};
/// n::Tensor features;
/// const auto scores = n::models::RevNetHead(c, x, 10, &features);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet(), RevContext
inline nn::Tensor RevNetHead(const RevContext &c, const RevPair &x, int classes,
                             nn::Tensor *features = nullptr) {
  if (classes < 2 || x.first.dim() != 4 || x.first.numel() == 0 ||
      x.first.shape() != x.second.shape())
    throw std::invalid_argument("invalid RevNet head dimensions");
  nn::Tensor h = RevNormRelu(c, "final/bn", nn::concat({x.first, x.second}, 3));
  if (features)
    *features = h;
  return nn::dense(c.p, "output", nn::global_avg_pool2d(h), classes);
}

/// @brief One coupling unit layout and optional ordinary-mode activation handles.
///
/// First unit in every stage is conservatively stored, including first-F
/// activation exceptions. Tensor pairs are populated only in ordinary mode.
///
/// @par Example
/// @code
/// const auto units = dm::nn::models::RevNetUnits(dm::nn::models::RevNetConfig::Cifar(38));
/// @endcode
/// @see RevNetUnits(), RevNetMode
struct RevNetUnit {
  std::string name;   ///< Stable stageS/unitU parameter prefix.
  int in;             ///< Full input width across both halves.
  int out;            ///< Full output width across both halves.
  int stride;         ///< Spatial stride, one or two.
  bool no_activation; ///< Skip first F preactivation in the model first unit.
  bool stored;        ///< Stage-entry input cannot be recovered by the standard inverse.
  RevPair input;      ///< Ordinary-mode input halves; undefined in other modes.
  RevPair output;     ///< Ordinary-mode output halves; undefined in other modes.
};
/// @brief Choose ordinary autograd, reconstructed activations or stored-input checkpointing.
///
/// Ordinary keeps graph activations. Reversible stores stage-entry inputs and
/// inverts other units; StoredUnits checkpoints each unit from stored inputs.
/// Recomputation costs arithmetic and suppresses repeated moving BN updates.
/// Floating reconstruction can perturb gradients; stored-input mode avoids that drift.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto cfg = n::models::RevNetConfig::Cifar(38);
/// const auto y = n::models::RevNet(p, state, n::ones({1, 8, 8, 3}), cfg);
/// const auto z = n::models::RevNet(p, state, n::ones({1, 8, 8, 3}), cfg, false,
///                                  n::models::RevNetMode::Reversible);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNet()
enum class RevNetMode {
  Ordinary,   ///< Keep ordinary autograd activations and fill unit input/output pairs.
  Reversible, ///< Checkpoint stage entries and reconstruct invertible stage interiors.
  StoredUnits ///< Checkpoint every unit, recomputing from its stored input.
};
/// @brief Stem, head features, logits and every unit layout from a RevNet call.
///
/// Handles retain storage and graphs. Unit activation pairs are populated only
/// in Ordinary mode; other modes retain only the documented checkpoint state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto cfg = n::models::RevNetConfig::Cifar(38);
/// const auto y = n::models::RevNet(p, state, n::ones({1, 8, 8, 3}), cfg);
/// @endcode
/// @see RevNet()
struct RevNetOutputs {
  nn::Tensor stem;               ///< Post-BN/ReLU stem, including initial pool for ImageNet.
  nn::Tensor features;           ///< Concatenated final BN/ReLU before global pooling.
  nn::Tensor logits;             ///< Unnormalised [N,classes] classifier scores.
  std::vector<RevNetUnit> units; ///< Ordered metadata and ordinary-mode activation pairs.
};

/// @brief Validate a RevNet layout and return stable unit names, dimensions and storage flags.
///
/// First unit of every stage is marked stored. Only the very first unit skips
/// F preactivation. Later units use stride one, input width equal to output.
/// Work/storage O(total units); no learned allocation or fitting.
///
/// @param cfg Configuration validated by Widths().
/// @return Independent metadata vector, with undefined input/output tensor pairs.
/// @throws std::invalid_argument For invalid configuration.
/// @par Example
/// @code
/// const auto units = dm::nn::models::RevNetUnits(dm::nn::models::RevNetConfig::Cifar(38));
/// @endcode
/// @see RevNetConfig, RevNetUnit
inline std::vector<RevNetUnit> RevNetUnits(const RevNetConfig &cfg) {
  const auto widths = cfg.Widths();
  std::vector<RevNetUnit> units;
  for (std::size_t stage = 0; stage < cfg.units.size(); ++stage)
    for (int unit = 0; unit < cfg.units[stage]; ++unit)
      units.push_back(
          {"stage" + std::to_string(stage + 1) + "/unit" + std::to_string(unit + 1),
           unit ? widths[stage + 1] : widths[stage],
           widths[stage + 1],
           unit ? 1 : cfg.strides[stage],
           stage == 0 && unit == 0,
           unit == 0,
           {},
           {}});
  return units;
}

/// @brief Build a complete reversible residual tensor classifier without fitting.
///
/// F/G use preactivated basic or bottleneck functions, with split additive
/// coupling. Stage entries store inputs; stage interiors can reconstruct inputs
/// from outputs. Bias-free SAME filters and learned-scale BN eps .001 follow
/// the release, with He-uniform fan-in replacing truncated-normal fan-out.
/// CIFAR uses 3x3 stem; ImageNet uses 7x7/2 and SAME max-pool. Head is BN/ReLU,
/// global mean and a biased dense classifier. No preprocessing, loss, optimizer
/// or fitting is hidden. Stores must outlive backward; do not modify weights
/// between forward/backward. Reversible callbacks borrow stores. Memory savings
/// concern activation storage, not parameters, optimizer state or process RSS.
///
/// @param p Mutable floating learnable store, creates/reuses named model leaves.
/// @param state Mutable BN moving moments matching p dtype.
/// @param image Nonempty floating NHWC images; basic downsampling requires even stage sides.
/// @param cfg Validated model layout, default CIFAR-110.
/// @param training Use batch BN moments when true, otherwise saved moments.
/// @param mode Ordinary, Reversible or StoredUnits; defaults Ordinary.
/// @param update Permit moving-moment updates in forward, default true; recomputation never updates.
/// @return Stem, final features, logits and every unit descriptor retaining gradients.
/// @throws std::invalid_argument For invalid layout, mode, geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto cfg = n::models::RevNetConfig::Cifar(38);
/// const auto y = n::models::RevNet(p, state, n::ones({1, 8, 8, 3}), cfg);
/// @endcode
/// @par References
/// - Gomez et al. (2017), https://arxiv.org/abs/1707.04585; https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
/// @see RevNetConfig, RevNetMode, RevBlockInverse()
inline RevNetOutputs RevNet(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const RevNetConfig &cfg = {},
                            bool training = false, RevNetMode mode = RevNetMode::Ordinary,
                            bool update = true) {
  if (image.dim() != 4 || image.numel() == 0 ||
      (mode != RevNetMode::Ordinary && mode != RevNetMode::Reversible &&
       mode != RevNetMode::StoredUnits))
    throw std::invalid_argument("invalid RevNet image or mode");
  const auto widths = cfg.Widths();
  RevNetOutputs out;
  const RevContext c{p, state, training, cfg.momentum, cfg.bottleneck, update};
  nn::Tensor h = RevNormRelu(
      c, "stem/bn",
      RevConv(p, "stem", image, widths[0], cfg.imagenet ? 7 : 3, cfg.imagenet ? 2 : 1));
  if (cfg.imagenet)
    h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
  out.stem = h;
  auto halves = nn::split(h, 2, 3);
  RevPair x{halves[0], halves[1]};
  out.units = RevNetUnits(cfg);
  for (std::size_t k = 0; k < out.units.size();) {
    auto &u = out.units[k];
    if (mode == RevNetMode::Reversible && !u.stored) {
      // Every following invertible unit of this stage in one reversible chain.
      std::vector<nn::ReversibleBlock> chain;
      for (; k < out.units.size() && !out.units[k].stored; ++k)
        chain.push_back(RevCoupling(c, out.units[k].name, out.units[k].out));
      auto y = nn::reversible_sequence(chain, x.first, x.second);
      x = {y[0], y[1]};
      continue;
    }
    // Stored (non-invertible) units of the reversible pass, and every unit of
    // StoredUnits, keep only their input and recompute during backward.
    if (mode != RevNetMode::Ordinary) {
      auto unit = [c, u](const std::vector<nn::Tensor> &in) {
        auto y =
            RevBlockForward(c, u.name, {in[0], in[1]}, u.in, u.out, u.stride, u.no_activation);
        return std::vector<nn::Tensor>{y.first, y.second};
      };
      auto y = nn::checkpoint(unit, {x.first, x.second});
      x = {y[0], y[1]};
    } else {
      u.input = x;
      x = RevBlockForward(c, u.name, x, u.in, u.out, u.stride, u.no_activation);
      u.output = x;
    }
    ++k;
  }
  out.logits = RevNetHead(c, x, cfg.classes, &out.features);
  return out;
}
} // namespace datamunge::nn::models
