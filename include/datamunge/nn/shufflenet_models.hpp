#pragma once
/// @file shufflenet_models.hpp
/// @brief Release/paper channel permutations and complete ShuffleNet-v1/v2 tensor classifiers.
#include "datamunge/nn/classic_cnn_components.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Reshape trailing channel dimensions while preserving the first NHW axes.
///
/// Also accepts an already split channel axis to flatten it again. Core reshape
/// checks element counts; input data is not mutated. Work/storage follows reshape.
///
/// @param x Nonempty tensor with rank at least four, any supported dtype.
/// @param channels Nonempty positive trailing dimensions with matching total element count.
/// @return Reshaped tensor retaining storage/autograd semantics of the core reshape.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto y = dm::nn::models::ReshapeChannels(dm::nn::ones({1, 2, 2, 12}), {4, 3});
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor ReshapeChannels(const nn::Tensor &x,
                                  const std::vector<std::int64_t> &channels) {
  if (x.dim() < 4 || x.numel() == 0 || channels.empty() ||
      std::any_of(channels.begin(), channels.end(), [](int64_t v) {
        return v < 1;
      }))
    throw std::invalid_argument("invalid channel reshape");
  nn::Shape shape(x.shape().begin(), x.shape().begin() + 3);
  shape.insert(shape.end(), channels.begin(), channels.end());
  return nn::reshape(x, shape);
}
/// @brief Permute channels using the official release shuffle order.
///
/// Input channel c moves to (c mod groups)*(C/groups)+floor(c/groups).
/// Implemented by reshape [C/groups,groups], transpose, flatten. This is
/// inverse to PaperChannelShuffle for the same group count. O(numel(x))
/// permutation work/storage; no learned parameters, randomness or fitting.
///
/// @param x Nonempty rank-four NHWC tensor.
/// @param channels Positive channel count matching x.size(3).
/// @param groups Positive divisor of channels; one returns x.
/// @return NHWC permutation, with exact inverse-permutation input gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto y =
///     dm::nn::models::ChannelShuffle(dm::nn::arange(0, 12).reshape({1, 1, 1, 12}), 12, 3);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor ChannelShuffle(const nn::Tensor &x, std::int64_t channels,
                                 std::int64_t groups) {
  if (x.dim() != 4 || x.numel() == 0 || channels < 1 || x.size(3) != channels || groups < 1 ||
      channels % groups)
    throw std::invalid_argument("channel shuffle needs channels divisible by groups");
  if (groups == 1)
    return x;
  return ReshapeChannels(
      nn::transpose(ReshapeChannels(x, {channels / groups, groups}), {0, 1, 2, 4, 3}),
      {channels});
}
/// @brief Permute channels with the paper reshape/groups transpose convention.
///
/// Reshape [groups,C/groups], transpose, flatten. This inverts the release
/// ChannelShuffle permutation and still mixes groups. The two are generally
/// different when groups!=C/groups. O(numel(x)); no learned state or fitting.
///
/// @param x Nonempty NHWC tensor.
/// @param channels Positive channel count matching input.
/// @param groups Positive divisor of channels.
/// @return Same-shaped channel permutation with inverse-permutation gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto y =
///     dm::nn::models::PaperChannelShuffle(dm::nn::arange(0, 12).reshape({1, 1, 1, 12}), 12, 3);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor PaperChannelShuffle(const nn::Tensor &x, std::int64_t channels,
                                      std::int64_t groups) {
  if (x.dim() != 4 || x.numel() == 0 || channels < 1 || x.size(3) != channels || groups < 1 ||
      channels % groups)
    throw std::invalid_argument("channel shuffle needs channels divisible by groups");
  return ReshapeChannels(
      nn::transpose(ReshapeChannels(x, {groups, channels / groups}), {0, 1, 2, 4, 3}),
      {channels});
}
/// @brief Pad both spatial axes of an NHWC tensor with a constant.
///
/// No mutation or fitting. Work/storage linear in padded element count. NaNs
/// in data propagate through copies; core padding converts value to input dtype.
///
/// @param x Rank-four tensor, positive channel dimension.
/// @param pad Nonnegative symmetric padding on height/width.
/// @param value Constant padding value, default zero; negative infinity is useful for max-pool.
/// @return Padded NHWC tensor retaining input derivatives.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto y = dm::nn::models::PadSpatial(dm::nn::ones({1, 3, 3, 2}), 1);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor PadSpatial(const nn::Tensor &x, std::int64_t pad, double value = 0) {
  if (x.dim() != 4 || x.numel() == 0 || pad < 0)
    throw std::invalid_argument("invalid spatial padding");
  return nn::pad(x, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}, nn::PadMode::Constant, value);
}
/// @brief Borrowed stores, BN mode and optional output-arithmetic log for ShuffleNet.
///
/// Referenced stores/log must outlive each synchronous call. Copies retain
/// references; callers synchronize shared mutable state. No fitting or owned optimizer.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ShuffleContext c{p, state, false};
/// @endcode
/// @see ShuffleConv()
struct ShuffleContext {
  nn::ParameterStore &p;     ///< Borrowed mutable floating learned store.
  nn::ParameterStore &state; ///< Borrowed moving BN store matching p dtype.
  bool training;             ///< Use batch moments when true, otherwise saved moments.
  double momentum = .9;      ///< Moving retention in [0,1).
  bool update_stats = true;  ///< Permit BN moving updates during training.
  std::vector<std::pair<nn::Shape, std::int64_t>> *layers =
      nullptr; ///< Optional borrowed log: output NHWC shape and MACs per output position.
};
/// @brief Apply dense/grouped/depthwise symmetric-padded convolution, BN and optional ReLU.
///
/// Bias-free He-uniform filters; learned-scale BN eps 1e-5. Grouped filters
/// retain per-group parameter names. Output log stores measured shape and
/// kernel²*Cin*Cout/groups or kernel²*Cin products per position. No fitting.
///
/// @param c Borrowed stores/BN mode and optional arithmetic log.
/// @param name Prefix for W or per-group groupN/W and /bn leaves.
/// @param x Nonempty floating NHWC, matching stores.
/// @param out Positive output width; depthwise requires out==Cin.
/// @param kernel Positive odd square side.
/// @param stride Positive spatial step.
/// @param groups Positive channel divisor for grouped pointwise; depthwise requires one.
/// @param depthwise True uses multiplier-one per-channel filters.
/// @param relu True applies ReLU after BN, false leaves BN linear.
/// @return NHWC differentiable output with explicit symmetric sampling geometry.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ShuffleContext c{p, state, false};
/// const auto y = n::models::ShuffleConv(c, "c", n::ones({1, 5, 5, 4}), 8, 3, 2, 1, false, true);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor ShuffleConv(ShuffleContext &c, const std::string &name, nn::Tensor x,
                              std::int64_t out, std::int64_t kernel, std::int64_t stride,
                              std::int64_t groups, bool depthwise, bool relu) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || kernel < 1 || kernel % 2 != 1 ||
      stride < 1 || groups < 1 || (depthwise && (out != x.size(3) || groups != 1)))
    throw std::invalid_argument("invalid shuffle convolution dimensions");
  const std::int64_t in = x.shape()[3];
  nn::ConvLayerOptions o;
  o.bias = false;
  o.init = nn::init::he_uniform();
  o.stride = stride;
  o.padding = nn::Padding::Valid;
  if (kernel > 1)
    x = PadSpatial(x, kernel / 2);
  nn::Tensor h = depthwise ? nn::depthwise_conv2d(c.p, name, x, 1, kernel, o)
                           : GroupedConv2D(c.p, name, x, out, kernel, groups, o);
  nn::BatchNormOptions bn;
  bn.momentum = c.momentum;
  bn.eps = 1e-5;
  bn.update_stats = c.update_stats;
  h = nn::batch_norm(c.p, c.state, name + "/bn", h, c.training, bn);
  if (c.layers)
    c.layers->push_back({h.shape(), kernel * kernel * (depthwise ? in : in * out / groups)});
  return relu ? nn::relu(h) : h;
}
/// @brief Average-pool a zero-padded NHWC map with 3x3 stride two.
///
/// One zero row/column pads each side; padding contributes to the divisor nine.
/// At an all-one corner the mean is 4/9. No state/fitting; O(numel(x)) work.
///
/// @param x Nonempty floating NHWC input.
/// @return NHWC map with sides ceil(H/2),ceil(W/2), retaining gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto y = dm::nn::models::AveragePoolWithPadding(dm::nn::ones({1, 5, 5, 2}));
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor AveragePoolWithPadding(const nn::Tensor &x) {
  return nn::avg_pool2d(PadSpatial(x, 1), 3, 2, nn::Padding::Valid);
}
/// @brief Apply grouped pointwise bottlenecks, release shuffle and a depthwise spatial filter.
///
/// The output/4 reduction uses BN/ReLU, then shuffle, linear depthwise/BN
/// and grouped linear expansion/BN. At stride two only the learned branch
/// is ReLU-activated before concatenation; input mean includes padding zeros.
/// No hidden fitting or post-concat permutation.
///
/// @param c Borrowed store/log and BN mode.
/// @param name Unit parameter prefix.
/// @param x Nonempty floating NHWC input.
/// @param out Positive width divisible by four, matching input for stride one.
/// @param groups Positive cardinality dividing inner/branch widths.
/// @param first Use an ungrouped initial reduction for the first model unit.
/// @param stride One or two; two concatenates pooled input with out-Cin branch channels.
/// @return Post-add ReLU at stride one or pooled-input/activated-branch concatenation.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ShuffleContext c{p, state, false};
/// const auto y = n::models::ShuffleV1Unit(c, "u", n::ones({1, 8, 8, 24}), 120, 3, true, 2);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor ShuffleV1Unit(ShuffleContext &c, const std::string &name,
                                const nn::Tensor &x, std::int64_t out, std::int64_t groups,
                                bool first, int stride) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || out % 4 || groups < 1 ||
      (stride != 1 && stride != 2) || (stride == 2 && out <= x.size(3)))
    throw std::invalid_argument("invalid ShuffleNet v1 unit dimensions");
  const std::int64_t in = x.shape()[3], mid = out / 4, branch = stride == 2 ? out - in : out;
  if (mid % groups || branch % groups || (stride == 1 && in != out))
    throw std::invalid_argument("invalid ShuffleNet v1 unit");
  nn::Tensor h =
      ShuffleConv(c, name + "/pointwise1", x, mid, 1, 1, first ? 1 : groups, false, true);
  h = ShuffleConv(c, name + "/depthwise", ChannelShuffle(h, mid, groups), mid, 3, stride, 1,
                  true, false);
  h = ShuffleConv(c, name + "/pointwise2", h, branch, 1, 1, groups, false, false);
  if (stride == 1)
    return nn::relu(h + x);
  return nn::concat({AveragePoolWithPadding(x), nn::relu(h)}, 3);
}
/// @brief Apply release-order v2 split/transform/concatenate without grouped pointwise filters.
///
/// Stride one extracts even input channels as carry and transforms odd channels
/// through pointwise/ReLU, depthwise/linear BN, pointwise/ReLU. Concatenate
/// carry then branch. Stride two uses the full input in both a depthwise/
/// pointwise projection keeping Cin channels and a branch producing out-Cin.
/// This follows the release ordering rather than a separate final paper shuffle.
///
/// @param c Borrowed stores/log and BN mode.
/// @param name Unit prefix.
/// @param x Nonempty floating NHWC input.
/// @param out Positive even output width, matching input and divisible by four at stride one.
/// @param stride One or two; stride two requires out>Cin.
/// @return NHWC map after even-channel carry plus transformed odd channels, or two strided paths.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ShuffleContext c{p, state, false};
/// const auto y = n::models::ShuffleV2Unit(c, "u", n::ones({1, 8, 8, 24}), 48, 2);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet(), ShuffleNetConfig
inline nn::Tensor ShuffleV2Unit(ShuffleContext &c, const std::string &name,
                                const nn::Tensor &x, std::int64_t out, int stride) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || out % 2 || (stride != 1 && stride != 2) ||
      (stride == 2 && out <= x.size(3)))
    throw std::invalid_argument("invalid ShuffleNet v2 unit dimensions");
  const std::int64_t in = x.shape()[3], mid = out / 2;
  if (stride == 1) {
    if (in != out || in % 4)
      throw std::invalid_argument("invalid ShuffleNet v2 unit");
    const auto parts = nn::split(ReshapeChannels(x, {in / 2, 2}), 2, 4);
    const auto even = ReshapeChannels(parts[0], {in / 2}),
               odd = ReshapeChannels(parts[1], {in / 2});
    nn::Tensor h = ShuffleConv(c, name + "/pointwise1", odd, mid, 1, 1, 1, false, true);
    h = ShuffleConv(c, name + "/depthwise", h, mid, 3, 1, 1, true, false);
    h = ShuffleConv(c, name + "/pointwise2", h, out - in / 2, 1, 1, 1, false, true);
    return nn::concat({even, h}, 3);
  }
  nn::Tensor projection =
      ShuffleConv(c, name + "/projection/depthwise", x, in, 3, 2, 1, true, false);
  projection =
      ShuffleConv(c, name + "/projection/pointwise", projection, in, 1, 1, 1, false, true);
  nn::Tensor h = ShuffleConv(c, name + "/pointwise1", x, mid, 1, 1, 1, false, true);
  h = ShuffleConv(c, name + "/depthwise", h, mid, 3, 2, 1, true, false);
  h = ShuffleConv(c, name + "/pointwise2", h, out - in, 1, 1, 1, false, true);
  return nn::concat({projection, h}, 3);
}
/// @brief Released ShuffleNet version, width preset, grouping and BN settings.
///
/// Default v2 1.0x has 1000 classes. Width preset metadata only; no parameter
/// allocation or fitting. Group count affects v1 only.
///
/// @par Example
/// @code
/// dm::nn::models::ShuffleNetConfig cfg;
/// const auto widths = cfg.Widths();
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleNet()
struct ShuffleNetConfig {
  int version = 2;    ///< Exactly one or two for the released model family.
  int groups = 3;     ///< V1 grouped pointwise cardinality, three or eight; ignored in v2.
  int classes = 1000; ///< At least two classifier outputs.
  std::string size = "1.0x"; ///< Released 0.5x/1.0x/1.5x/2.0x preset name.
  bool small_stem = false;   ///< Stem stride one and no initial max-pool for small images.
  double momentum = .9;      ///< BN moving retention in [0,1).
  bool update_stats = true;  ///< Permit moving updates during training.

  /// @brief Validate settings and return released stem/stage/final widths.
  ///
  /// V1 supports group-three/eight tables; v2 ignores groups. Returns independent
  /// metadata and does not mutate model state. O(1) preset lookup/copy.
  ///
  /// @return Four v1 widths or five v2 widths, the last a final 1x1 output width.
  /// @throws std::invalid_argument For invalid version, v1 groups, size name, classes or momentum.
  /// @par Example
  /// @code
  /// const auto w = dm::nn::models::ShuffleNetConfig{}.Widths(); // {24,116,232,464,1024}
  /// @endcode
  /// @see ShuffleNet()
  std::vector<int> Widths() const {
    /// @brief Released v1 group-three stem/stage widths by size preset.
    static const std::map<std::string, std::vector<int>> v1g3 = {
        {"0.5x", {12, 120, 240, 480}},
        {"1.0x", {24, 240, 480, 960}},
        {"1.5x", {24, 360, 720, 1440}},
        {"2.0x", {48, 480, 960, 1920}}};
    /// @brief Released v1 group-eight stem/stage widths by size preset.
    static const std::map<std::string, std::vector<int>> v1g8 = {
        {"0.5x", {16, 192, 384, 768}},
        {"1.0x", {24, 384, 768, 1536}},
        {"1.5x", {24, 576, 1152, 2304}},
        {"2.0x", {48, 768, 1536, 3072}}};
    /// @brief Released v2 stem/stage/final widths by size preset.
    static const std::map<std::string, std::vector<int>> v2 = {
        {"0.5x", {24, 48, 96, 192, 1024}},
        {"1.0x", {24, 116, 232, 464, 1024}},
        {"1.5x", {24, 176, 352, 704, 1024}},
        {"2.0x", {24, 244, 488, 976, 2048}}};
    if (version != 1 && version != 2)
      throw std::invalid_argument("invalid ShuffleNet configuration");
    if (version == 1 && groups != 3 && groups != 8)
      throw std::invalid_argument("ShuffleNet v1 releases use 3 or 8 groups");
    const auto &table = version == 2 ? v2 : groups == 3 ? v1g3 : v1g8;
    if (classes < 2 || !table.count(size) || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid ShuffleNet configuration");
    return table.at(size);
  }
};
/// @brief Final features/logits, three stages and measured convolution metadata.
///
/// Tensor handles retain storage/graphs, metadata copies shapes/counts. Learned
/// leaves and BN state remain in caller stores. Per-image MACs sum H*W*count
/// over layers plus classifier_macs, excluding BN, activation, pooling, permutation
/// and addition. Counts do not measure latency.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ShuffleNetConfig cfg;
/// cfg.size = "0.5x";
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::ShuffleNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @see ShuffleNet()
struct ShuffleNetOutputs {
  nn::Tensor features;            ///< Final stage map (v1) or final 1x1/BN/ReLU map (v2).
  nn::Tensor logits;              ///< Unnormalised [N,classes] bias-free dense scores.
  std::vector<nn::Tensor> stages; ///< Three complete stage ends, 4/8/4 units.
  std::vector<std::pair<nn::Shape, std::int64_t>>
      layers; ///< Convolution output shapes and kernel MACs per output position.
  std::int64_t classifier_macs = 0; ///< Final classifier input width times classes, per image.
};
/// @brief Build complete released ShuffleNet-v1/v2 tensor logits without fitting.
///
/// Three stages contain 4/8/4 units, downsampling at their first units.
/// Standard stem uses 3x3/2 conv/BN/ReLU and padded max-pool; small_stem
/// uses stride one and drops initial pooling while retaining every stage.
/// V2 appends a final 1x1/BN/ReLU and uses .2 dropout for 2.0x only, drawn
/// from p.generator(). Both globally average and use a bias-free Glorot
/// classifier. He-uniform fan-in is an initialization adaptation; BN learns
/// scale/shift with epsilon 1e-5. No preprocessing, optimizer or fitting.
/// Counts describe kernel arithmetic, not measured latency.
///
/// @param p Mutable floating learnable store, lazily creates/reuses leaves.
/// @param state Mutable BN moving moments, same dtype as p.
/// @param image Nonempty floating NHWC input.
/// @param cfg Valid version, group and size preset, default v2 1.0x ImageNet.
/// @param training True uses batch BN moments and optional v2 2.0x dropout.
/// @return Three stages, final features/logits and measured arithmetic metadata with gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ShuffleNetConfig cfg;
/// cfg.size = "0.5x";
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::ShuffleNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Zhang et al. (2018), https://arxiv.org/abs/1707.01083; Ma et al. (2018), https://arxiv.org/abs/1807.11164; https://github.com/megvii-model/ShuffleNet-Series
/// @see ShuffleV1Unit(), ShuffleV2Unit()
inline ShuffleNetOutputs ShuffleNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &image, const ShuffleNetConfig &cfg = {},
                                    bool training = false) {
  if (image.dim() != 4 || image.numel() == 0)
    throw std::invalid_argument("ShuffleNet requires nonempty NHWC images");
  const auto widths = cfg.Widths();
  ShuffleNetOutputs out;
  ShuffleContext c{p, state, training, cfg.momentum, cfg.update_stats, &out.layers};
  nn::Tensor h =
      ShuffleConv(c, "conv0", image, widths[0], 3, cfg.small_stem ? 1 : 2, 1, false, true);
  if (!cfg.small_stem)
    h = nn::max_pool2d(PadSpatial(h, 1, -std::numeric_limits<double>::infinity()), 3, 2,
                       nn::Padding::Valid);
  std::int64_t in = widths[0];
  const int repeats[3] = {4, 8, 4};
  for (int stage = 0; stage < 3; ++stage) {
    const int width = widths[stage + 1];
    for (int unit = 0; unit < repeats[stage]; ++unit, in = width) {
      const auto name =
          "stage" + std::to_string(stage + 2) + "/unit" + std::to_string(unit + 1);
      h = cfg.version == 1 ? ShuffleV1Unit(c, name, h, width, cfg.groups,
                                           stage == 0 && unit == 0, unit ? 1 : 2)
                           : ShuffleV2Unit(c, name, h, width, unit ? 1 : 2);
    }
    out.stages.push_back(h);
  }
  if (cfg.version == 2) {
    h = ShuffleConv(c, "conv_last", h, widths[4], 1, 1, 1, false, true);
    in = widths[4];
  }
  out.features = h;
  nn::Tensor pooled = nn::global_avg_pool2d(h);
  if (training && cfg.version == 2 && cfg.size == "2.0x")
    pooled = nn::dropout(pooled, .2, true, p.generator());
  nn::DenseOptions classifier;
  classifier.bias = false;
  out.logits = nn::dense(p, "output", pooled, cfg.classes, classifier);
  out.classifier_macs = in * cfg.classes;
  return out;
}
} // namespace datamunge::nn::models
