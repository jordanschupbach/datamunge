#pragma once
/// @file dual_path_models.hpp
/// @brief Coupled residual-addition and dense-concatenation tensor vision models; no fitting.
#include "datamunge/nn/preactivation_densenet_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Reference stem, bottleneck, cardinality and per-stage dual-path metadata.
///
/// Integer architecture metadata only; no tensors or fitted state. Each stage
/// starts with one projection unit. Group count is not implicitly width-scaled.
///
/// @par Example
/// @code
/// const auto layout = dm::nn::models::DPNLayoutFor(92);
/// @endcode
/// @par References
/// - Chen et al. (2017), Dual Path Networks, https://arxiv.org/abs/1707.01629; released model implementation, https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
/// @see DPNLayoutFor(), DPNConfig
struct DPNLayout {
  bool small;                ///< DPN-68 uses smaller outer stage widths and a 3x3 stem.
  int init;                  ///< Reference stem output channels.
  int k_r;                   ///< First-stage bottleneck width before user scaling.
  int groups;                ///< Released grouped 3x3 cardinality.
  std::array<int, 4> blocks; ///< Unit repetitions in stages one through four.
  std::array<int, 4> inc;    ///< New dense channels per unit before user scaling.
};
/// @brief Return reference layout metadata for a released DPN depth.
///
/// Depth is three convolutions per unit plus stem/classifier; stage projections
/// are excluded from that depth label. No model is allocated or fitted.
///
/// @param depth Exactly 68, 92, 98, 107 or 131.
/// @return Independent value containing the released widths, cardinality and four-stage counts.
/// @throws std::invalid_argument For any unsupported depth.
/// @par Example
/// @code
/// const auto layout = dm::nn::models::DPNLayoutFor(92); // 3,4,20,3 units
/// @endcode
/// @par References
/// - Chen et al. (2017), Dual Path Networks, https://arxiv.org/abs/1707.01629; released model implementation, https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
/// @see DPN()
inline DPNLayout DPNLayoutFor(int depth) {
  switch (depth) {
  case 68:
    return {true, 10, 128, 32, {3, 4, 12, 3}, {16, 32, 32, 64}};
  case 92:
    return {false, 64, 96, 32, {3, 4, 20, 3}, {16, 32, 24, 128}};
  case 98:
    return {false, 96, 160, 40, {3, 6, 20, 3}, {16, 32, 32, 128}};
  case 107:
    return {false, 128, 200, 50, {4, 8, 20, 3}, {20, 64, 64, 128}};
  case 131:
    return {false, 128, 160, 40, {4, 8, 28, 3}, {16, 32, 32, 128}};
  default:
    throw std::invalid_argument("supported DPN depths are 68/92/98/107/131");
  }
}

/// @brief Fixed-width residual and growing dense tensors, with explicit channel metadata.
///
/// A stem has no dense channels and may leave dense undefined. Initialized
/// residual/dense maps must share NHW and dtype, with metadata matching their
/// last axes. Default empty paths are placeholders until assigned.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::DPNPaths x{n::ones({1, 5, 5, 4}), {}, 4, 0};
/// @endcode
/// @see DPNJoin(), DPNBlock()
struct DPNPaths {
  nn::Tensor residual; ///< Floating NHWC residual representation, positive channels.
  nn::Tensor dense;    ///< Floating NHWC dense features; ignored when dense_channels is zero.
  std::int64_t residual_channels = 0; ///< Actual positive residual width once initialized.
  std::int64_t dense_channels = 0;    ///< Actual nonnegative dense width once initialized.

  /// @brief Sum residual and dense channel metadata without reading tensor values.
  ///
  /// Does not validate tensors; DPNJoin checks initialized paths before use.
  ///
  /// @return residual_channels+dense_channels; zero for default uninitialized paths.
  /// @par Example
  /// @code
  /// const auto width =
  ///     dm::nn::models::DPNPaths{dm::nn::ones({1, 2, 2, 4}), {}, 4, 0}.channels(); // 4
  /// @endcode
  /// @see DPNJoin()
  std::int64_t channels() const {
    return residual_channels + dense_channels;
  }
};
/// @brief Concatenate residual then dense channels, or return a stem residual unchanged.
///
/// Validates channel metadata. With a dense path, core concat checks common
/// NHW/dtype. Inputs are not mutated; no fitted state is created.
///
/// @param x Initialized paths with positive residual and nonnegative dense width metadata.
/// @return Floating NHWC joined tensor, with graph retained; residual handle if dense width is zero.
/// @throws std::invalid_argument For invalid metadata/tensors or incompatible concatenation.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto joined = n::models::DPNJoin({n::ones({1, 3, 3, 4}), {}, 4, 0});
/// @endcode
/// @see DPNPaths, DPNBlock()
inline nn::Tensor DPNJoin(const DPNPaths &x) {
  if (x.residual_channels <= 0 || x.dense_channels < 0 || x.residual.dim() != 4 ||
      x.residual.size(3) != x.residual_channels ||
      (x.dense_channels && (x.dense.dim() != 4 || x.dense.size(3) != x.dense_channels)))
    throw std::invalid_argument("DPN channel metadata must match initialized paths");
  return x.dense_channels ? nn::concat({x.residual, x.dense}, 3) : x.residual;
}
/// @brief Select a contiguous channel range from a dual-path feature map.
///
/// Does not mutate x or create learned state. A zero count gives an empty channel range.
///
/// @param x Floating NHWC map.
/// @param start Nonnegative first channel.
/// @param count Nonnegative retained width, with start+count at most x.size(3).
/// @return Differentiable NHWC channel slice in original order; storage follows core slice semantics.
/// @throws std::invalid_argument For invalid rank or channel bounds.
/// @par Example
/// @code
/// const auto y = dm::nn::models::DPNChannelSlice(dm::nn::ones({1, 3, 3, 4}), 1, 2);
/// @endcode
/// @see DPNJoin()
inline nn::Tensor DPNChannelSlice(const nn::Tensor &x, std::int64_t start,
                                  std::int64_t count) {
  if (x.dim() != 4 || start < 0 || count < 0 || start > x.size(3) || count > x.size(3) - start)
    throw std::invalid_argument("invalid DPN channel slice");
  return nn::slice_axis(x, 3, start, start + count);
}

/// @brief Apply learned BN/ReLU with epsilon .001, then a raw optionally grouped convolution.
///
/// One fused grouped HWIO kernel stores [k,k,Cin/groups,Cout], preserving
/// per-group fan-in/count but differing from the older per-group tensor names.
/// No fitting occurs; moving moments update eagerly only as norm requests.
///
/// @param p Mutable floating learnable store, name/W plus BN gamma/beta.
/// @param state Distinct moving moments, same dtype as p.
/// @param name Prefix; preactivation BN uses name/bn.
/// @param input Nonempty floating NHWC tensor matching stores.
/// @param out Positive output channels divisible by groups.
/// @param kernel Positive square kernel width; normal DPN layers use one or three.
/// @param norm Preactivation mode, momentum and moving-update flag; epsilon is overridden to .001.
/// @param stride Positive common spatial stride, default one.
/// @param groups Positive input/output channel divisor, default one.
/// @return Floating NHWC raw filter output with retained graph, without output activation.
/// @throws std::invalid_argument For invalid tensor/group geometry, BN settings, dtype or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PreactivationNormConfig norm;
/// const auto y =
///     n::models::DPNPreActConv(p, state, "c", n::ones({2, 5, 5, 4}), 4, 3, norm, 1, 2);
/// @endcode
/// @par References
/// - Chen et al. (2017), Dual Path Networks, https://arxiv.org/abs/1707.01629; released model implementation, https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
/// @see RawResidualConv(), DPNBlock()
inline nn::Tensor DPNPreActConv(nn::ParameterStore &p, nn::ParameterStore &state,
                                const std::string &name, const nn::Tensor &input,
                                std::int64_t out, std::int64_t kernel,
                                const PreactivationNormConfig &norm, std::int64_t stride = 1,
                                std::int64_t groups = 1) {
  PreactivationNormConfig bn = norm;
  bn.epsilon = 1e-3; // the released DPN BN epsilon, whatever the caller's default
  const nn::Tensor h = PreActivate(p, state, name + "/bn", input, bn);
  return RawResidualConv(p, name, h, out, kernel, stride, groups);
}

/// @brief Projected input paths, learned updates and resulting dual-path representation.
///
/// input refers to paths after the optional stage projection, not necessarily
/// the caller's original tensors. All handles retain storage and gradient graphs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const n::models::DPNPaths x{n::ones({2, 5, 5, 4}), {}, 4, 0};
/// const auto y = n::models::DPNBlock(p, state, "b", x, 4, 4, 1, 2, true);
/// @endcode
/// @see DPNBlock()
struct DPNBlockOutputs {
  DPNPaths input;             ///< Paths after optional projection, before the branch updates.
  DPNPaths output;            ///< Fixed-width residual sum and dense concatenation.
  nn::Tensor residual_update; ///< First width channels of the shared bottleneck output.
  nn::Tensor dense_update;    ///< Last inc new channels of the shared bottleneck output.
};
/// @brief Split a shared preactivation bottleneck into residual addition and dense concatenation.
///
/// Projection initializes width residual channels and 2*inc dense channels.
/// The shared branch is preactivated 1x1, grouped 3x3 (the downsample), then
/// 1x1 to width+inc. First width channels add to residual; last inc append to
/// dense. Ordinary units require matching residual width and an existing dense
/// path. Signed old paths pass unchanged when both branch updates are zero.
/// No fitting occurs; convolution plus pointwise work dominates.
///
/// @param p Mutable floating learnable store.
/// @param state Distinct BN moving moments, matching p dtype.
/// @param name Prefix for projection and conv1/2/3 layers.
/// @param x Initialized incoming paths, joined in residual-then-dense order.
/// @param width Positive fixed residual output width.
/// @param bottleneck Positive inner width divisible by groups.
/// @param inc Positive new dense channels per unit.
/// @param groups Positive grouped 3x3 cardinality dividing bottleneck.
/// @param project True starts a stage with a separate BN/ReLU/1x1 projection to width+2*inc.
/// @param stride One (default) or two; downsampling requires project=true.
/// @param norm BN mode/momentum/update settings; every layer forces epsilon .001.
/// @return Projected input, branch updates and output paths; no activation follows the path combination.
/// @throws std::invalid_argument For invalid path metadata, dimensions/cardinality, stride, BN or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const n::models::DPNPaths x{n::ones({2, 5, 5, 4}), {}, 4, 0};
/// const auto y = n::models::DPNBlock(p, state, "b", x, 4, 4, 1, 2, true);
/// @endcode
/// @par References
/// - Chen et al. (2017), Dual Path Networks, https://arxiv.org/abs/1707.01629; released model implementation, https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
/// @see DPNPaths, DPN()
inline DPNBlockOutputs DPNBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                const std::string &name, const DPNPaths &x, std::int64_t width,
                                std::int64_t bottleneck, std::int64_t inc, std::int64_t groups,
                                bool project, int stride = 1,
                                const PreactivationNormConfig &norm = {}) {
  if (width < 1 || bottleneck < 1 || inc < 1 || groups < 1 || bottleneck % groups ||
      (stride != 1 && stride != 2) || (!project && stride != 1) ||
      (!project && (x.residual_channels != width || !x.dense_channels)))
    throw std::invalid_argument("invalid DPN unit dimensions");
  const nn::Tensor joined = DPNJoin(x);
  DPNBlockOutputs out;
  out.input = x;
  if (project) {
    const nn::Tensor projection = DPNPreActConv(p, state, name + "/projection", joined,
                                                width + 2 * inc, 1, norm, stride);
    out.input = {DPNChannelSlice(projection, 0, width),
                 DPNChannelSlice(projection, width, 2 * inc), width, 2 * inc};
  }
  nn::Tensor h = DPNPreActConv(p, state, name + "/conv1", joined, bottleneck, 1, norm);
  h = DPNPreActConv(p, state, name + "/conv2", h, bottleneck, 3, norm, stride, groups);
  h = DPNPreActConv(p, state, name + "/conv3", h, width + inc, 1, norm);
  out.residual_update = DPNChannelSlice(h, 0, width);
  out.dense_update = DPNChannelSlice(h, width, inc);
  out.output = {out.input.residual + out.residual_update,
                nn::concat({out.input.dense, out.dense_update}, 3), width,
                out.input.dense_channels + inc};
  return out;
}

/// @brief DPN release depth, exact width/increment scaling, group override and stem/BN settings.
///
/// Cardinality zero selects the full release value; it is not automatically
/// scaled when widths shrink. For compact DPN-92, divisor four and groups eight
/// make the first 24-wide bottleneck divisible. All block repetitions remain.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DPNConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.groups = 8;
/// cfg.small_stem = true;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Chen et al. (2017), Dual Path Networks, https://arxiv.org/abs/1707.01629; released model implementation, https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
/// @see DPN()
struct DPNConfig {
  int depth = 92;     ///< Reference layout: 68,92,98,107,131.
  int classes = 1000; ///< At least two output scores.
  int width_divisor =
      1; ///< Positive exact divisor of stem, every outer width and dense increment.
  int groups =
      0; ///< Zero uses release cardinality; otherwise a positive divisor of every bottleneck.
  bool small_stem = false;  ///< True uses 3x3 stride one without pooling for small images.
  double momentum = .9;     ///< Finite BN old-value EMA retention in [0,1).
  bool update_stats = true; ///< Update moving moments only during training and when true.

  /// @brief Validate depth, widths, increments, cardinality, classes and BN momentum.
  ///
  /// Checks integer metadata before allocating kernels; input geometry is checked
  /// by the forward builder. No configuration fields are changed.
  ///
  /// @throws std::invalid_argument For unsupported depth or incompatible dimensions/cardinality/probabilities.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// n::ParameterStore p(17), state(17);
  /// n::models::DPNConfig cfg;
  /// cfg.classes = 10;
  /// cfg.width_divisor = 4;
  /// cfg.groups = 8;
  /// cfg.small_stem = true;
  /// cfg.Validate();
  /// @endcode
  /// @see DPNLayoutFor()
  void Validate() const {
    const auto layout = DPNLayoutFor(depth);
    const int divisor = width_divisor, cardinality = groups ? groups : layout.groups,
              factor = layout.small ? 1 : 4;
    if (divisor < 1 || layout.init % divisor || cardinality < 1 || classes < 2 ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid DPN configuration");
    for (int stage = 0; stage < 4; ++stage) {
      const int full_width = (64 << stage) * factor;
      if (full_width % divisor || layout.inc[stage] % divisor)
        throw std::invalid_argument("DPN width divisor must divide every width/increment");
      const std::int64_t bottleneck =
          std::int64_t(layout.k_r) * (full_width / divisor) / (64 * factor);
      if (bottleneck < 1 || bottleneck % cardinality)
        throw std::invalid_argument("DPN bottleneck must divide by groups");
    }
  }
};
/// @brief Stem, four dual-path stages, every unit and final classifier outputs.
///
/// Handles retain storage and gradient graphs. Fitted leaves and moving moments
/// remain in the caller-owned stores.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DPNConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.groups = 8;
/// cfg.small_stem = true;
/// const auto y = n::models::DPN(p, state, n::zeros({1, 16, 16, 3}), cfg);
/// @endcode
/// @see DPN()
struct DPNOutputs {
  nn::Tensor stem;     ///< Initial convolution/BN/ReLU and optional max-pool map.
  nn::Tensor features; ///< Joined final paths after BN/ReLU, before global averaging.
  nn::Tensor logits;   ///< Unnormalised [N,classes] linear scores.
  std::array<DPNPaths, 4> stages; ///< Final residual/dense paths for each stage.
  std::vector<std::vector<DPNBlockOutputs>> blocks; ///< Every unit grouped by stage.
};
/// @brief Build a complete released-depth dual-path network without fitting.
///
/// Every stage starts a projected dual path, then grows dense width by inc per
/// unit while residual width stays fixed. Complete unit counts are retained at
/// scaled widths. Standard stem is strided with padded max-pool; DPN-68 uses
/// a 3x3 kernel, other releases 7x7. small_stem removes downsampling/pool.
/// All convolutions are bias-free He initialized; learned-scale BN epsilon is
/// .001 with unbiased EMA moving variance. Final BN/ReLU, GAP and a biased
/// linear classifier follow. No dropout, pretrained weights or optimizer is hidden.
///
/// @param p Mutable floating learnable store.
/// @param state Distinct BN moving moments, matching p dtype.
/// @param image Nonempty floating NHWC RGB batch.
/// @param cfg Valid released layout, exact dimensions/cardinality, stem and BN settings.
/// @param training Use batch moments with optional eager moving updates when true; false by default.
/// @return Stem, all units/stages, final features and [N,classes] logits retaining their graphs.
/// @throws std::invalid_argument For invalid cfg, RGB/path geometry, dtype, BN or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DPNConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.groups = 8;
/// cfg.small_stem = true;
/// const auto y = n::models::DPN(p, state, n::zeros({1, 16, 16, 3}), cfg);
/// @endcode
/// @par References
/// - Chen et al. (2017), Dual Path Networks, https://arxiv.org/abs/1707.01629; released model implementation, https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py
/// @see DPNBlock(), DPNConfig
inline DPNOutputs DPN(nn::ParameterStore &p, nn::ParameterStore &state,
                      const nn::Tensor &image, const DPNConfig &cfg = {},
                      bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(0) <= 0 || image.size(1) <= 0 || image.size(2) <= 0 ||
      image.size(3) != 3)
    throw std::invalid_argument("nonempty NHWC RGB DPN input required");
  const auto layout = DPNLayoutFor(cfg.depth);
  const int divisor = cfg.width_divisor, groups = cfg.groups ? cfg.groups : layout.groups,
            factor = layout.small ? 1 : 4;
  if (divisor < 1 || layout.init % divisor || groups < 1 || cfg.classes < 2 ||
      !(cfg.momentum >= 0 && cfg.momentum < 1))
    throw std::invalid_argument("invalid DPN configuration");
  DPNOutputs out;
  const PreactivationNormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-3};
  ResNetNormConfig stem_norm;
  stem_norm.training = training;
  stem_norm.scale = true;
  stem_norm.epsilon = 1e-3;
  stem_norm.momentum = cfg.momentum;
  stem_norm.update_stats = cfg.update_stats;
  const std::int64_t init = layout.init / divisor;
  nn::Tensor h =
      ResNetConv(p, state, "stem", image, init, cfg.small_stem || layout.small ? 3 : 7,
                 stem_norm, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  DPNPaths x = {h, {}, init, 0};
  for (int stage = 0; stage < 4; ++stage) {
    const int full_width = (64 << stage) * factor;
    if (full_width % divisor || layout.inc[stage] % divisor)
      throw std::invalid_argument("DPN width divisor must divide every width");
    const std::int64_t width = full_width / divisor, inc = layout.inc[stage] / divisor;
    const std::int64_t bottleneck = layout.k_r * width / (64 * factor);
    out.blocks.emplace_back();
    for (int block = 0; block < layout.blocks[stage]; ++block) {
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      out.blocks.back().push_back(DPNBlock(p, state, name, x, width, bottleneck, inc, groups,
                                           block == 0, stage > 0 && block == 0 ? 2 : 1, norm));
      x = out.blocks.back().back().output;
    }
    out.stages[stage] = x;
  }
  out.features = PreActivate(p, state, "final/bn", DPNJoin(x), norm);
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
