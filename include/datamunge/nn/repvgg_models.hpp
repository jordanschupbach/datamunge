#pragma once
/// @file repvgg_models.hpp
/// @brief Tensor RepVGG multi-branch models and dtype-preserving explicit inference fusion.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/batch_statistics.hpp"
#include "datamunge/nn/classic_cnn_components.hpp"
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Released RepVGG stage repeats, width multipliers and grouping metadata.
///
/// Value metadata, no learned state or fitting.
///
/// @par Example
/// @code
/// const auto layout = dm::nn::models::RepVGGLayout("A0");
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGG(), RepVGGBlock(), FuseRepVGGBlock()
struct RepVGGModel {
  std::vector<int> blocks;   ///< Four positive stage repeat counts, excluding stem.
  std::vector<double> width; ///< Four released positive stage width multipliers.
  int groups;                ///< One, two or four, applied to even blocks 2..26.
};
/// @brief Return one of ten released RepVGG architecture presets.
///
/// Immutable local lookup is initialized once; concurrent reads are safe.
/// A stages repeat 2/4/14/1; B repeats 4/6/16/1. No model allocation/search.
///
/// @param name A0,A1,A2,B0,B1,B1g2,B1g4,B2,B2g4 or B3, case-sensitive.
/// @return Owned stage counts/width multipliers and group metadata.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @par Example
/// @code
/// const auto layout = dm::nn::models::RepVGGLayout("B1g4");
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline RepVGGModel RepVGGLayout(const std::string &name) {
  /// @brief Immutable ten-preset release table, copied to callers.
  static const std::map<std::string, RepVGGModel> models = {
      {"A0", {{2, 4, 14, 1}, {.75, .75, .75, 2.5}, 1}},
      {"A1", {{2, 4, 14, 1}, {1, 1, 1, 2.5}, 1}},
      {"A2", {{2, 4, 14, 1}, {1.5, 1.5, 1.5, 2.75}, 1}},
      {"B0", {{4, 6, 16, 1}, {1, 1, 1, 2.5}, 1}},
      {"B1", {{4, 6, 16, 1}, {2, 2, 2, 4}, 1}},
      {"B1g2", {{4, 6, 16, 1}, {2, 2, 2, 4}, 2}},
      {"B1g4", {{4, 6, 16, 1}, {2, 2, 2, 4}, 4}},
      {"B2", {{4, 6, 16, 1}, {2.5, 2.5, 2.5, 5}, 1}},
      {"B2g4", {{4, 6, 16, 1}, {2.5, 2.5, 2.5, 5}, 4}},
      {"B3", {{4, 6, 16, 1}, {3, 3, 3, 5}, 1}}};
  if (!models.count(name))
    throw std::invalid_argument("unknown RepVGG model " + name);
  return models.at(name);
}

/// @brief Validated geometry and learned prefix for one RepVGG block.
///
/// Value metadata, no learned state or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::RepVGGBlockSpec b{"b", 8, 8, 1, 2, true};
/// b.Validate();
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGG(), RepVGGBlock(), FuseRepVGGBlock()
struct RepVGGBlockSpec {
  std::string name; ///< Prefix for dense/pointwise/identity or reparam leaves.
  int in;           ///< Positive input width divisible by groups.
  int out;          ///< Positive output width divisible by groups.
  int stride;       ///< Positive common branch and shortcut spatial step.
  int groups;       ///< Positive divisor of input/output widths.
  bool identity;    ///< Permit input BN branch only at equal widths and stride one.

  /// @brief Reject invalid grouped block geometry or an impossible identity branch.
  ///
  /// Architecture metadata validation only; leaves stores untouched.
  ///
  /// @throws std::invalid_argument For nonpositive widths/stride/groups, inexact groups, or incompatible identity.
  /// @par Example
  /// @code
  /// dm::nn::models::RepVGGBlockSpec{"b", 8, 8, 1, 2, true}.Validate();
  /// @endcode
  /// @see RepVGGBlock()
  void Validate() const {
    if (in < 1 || out < 1 || stride < 1 || groups < 1 || in % groups || out % groups ||
        (identity && (in != out || stride != 1)))
      throw std::invalid_argument("invalid RepVGG block geometry");
  }
};
/// @brief Final maps/logits and every executed RepVGG block specification.
///
/// Tensor handles own storage references and retain any active differentiation
/// graphs. Parameter/state stores retain source learned leaves; no fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RepVGGConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::RepVGG(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGG(), RepVGGBlock(), FuseRepVGGBlock()
struct RepVGGOutputs {
  nn::Tensor features;                 ///< Last post-ReLU NHWC map.
  nn::Tensor logits;                   ///< Global-mean classifier [N,classes] scores.
  std::vector<RepVGGBlockSpec> blocks; ///< Stem and every stage block in execution order.
  std::vector<nn::Tensor> outputs;     ///< Corresponding post-ReLU maps retaining gradients.
};

/// @brief Construct the learned-scale BN options used by every training branch.
///
/// Options are value metadata; no state or fitting.
///
/// @param momentum Finite moving retention in [0,1).
/// @param update_stats Permit eager training moving-moment updates.
/// @return BatchNormOptions with epsilon 1e-5, scale/center enabled.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @par Example
/// @code
/// const auto options = dm::nn::models::RepVGGNorm(.9, false);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline nn::BatchNormOptions RepVGGNorm(double momentum, bool update_stats) {
  if (!(momentum >= 0 && momentum < 1))
    throw std::invalid_argument("invalid RepVGG BN momentum");
  nn::BatchNormOptions bn;
  bn.momentum = momentum;
  bn.eps = 1e-5;
  bn.update_stats = update_stats;
  return bn;
}

/// @brief Apply one bias-free grouped filter followed by learned-scale BN.
///
/// He-uniform independent per-group filters, no bias, BN eps 1e-5.
/// No branch activation; the block adds all branches before applying ReLU.
/// No fitting; caller synchronizes shared learned/state stores.
///
/// @param p Mutable floating learned store; per-group named filters lazily created.
/// @param state Mutable matching-dtype BN moving moments.
/// @param name Dense/pointwise branch prefix.
/// @param x Nonempty floating NHWC input matching stores.
/// @param out Positive output width divisible by groups.
/// @param kernel One or three, symmetric pad for the spatial branch.
/// @param stride Positive square spatial step.
/// @param groups Positive divisor of input/output widths.
/// @param training Use batch BN when true, moving values when false.
/// @param momentum Moving retention in [0,1).
/// @param update_stats Permit eager training moving updates, default true.
/// @return Linear NHWC branch map before the block sum/ReLU.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y =
///     n::models::RepVGGBranch(p, state, "b", n::ones({2, 5, 5, 8}), 8, 3, 1, 2, false, .9);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline nn::Tensor RepVGGBranch(nn::ParameterStore &p, nn::ParameterStore &state,
                               const std::string &name, const nn::Tensor &x, int out,
                               int kernel, int stride, int groups, bool training,
                               double momentum, bool update_stats = true) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || (kernel != 1 && kernel != 3) ||
      stride < 1 || groups < 1 || x.size(3) % groups || out % groups)
    throw std::invalid_argument("invalid RepVGG branch geometry");
  nn::ConvLayerOptions cfg;
  cfg.bias = false;
  cfg.init = nn::init::he_uniform();
  cfg.stride = stride;
  cfg.padding = nn::Padding::Valid;
  const nn::Tensor in =
      kernel > 1
          ? nn::pad(x, {{0, 0}, {kernel / 2, kernel / 2}, {kernel / 2, kernel / 2}, {0, 0}})
          : x;
  const nn::Tensor h = GroupedConv2D(p, name, in, out, kernel, groups, cfg);
  return nn::batch_norm(p, state, name + "/bn", h, training,
                        RepVGGNorm(momentum, update_stats));
}

/// @brief Apply a multi-branch training-form or single-filter deployment block.
///
/// Training form sums 3x3/BN plus 1x1/BN and optional identity/BN.
/// Symmetric pad one aligns the strided 3x3 centres with the unpadded
/// 1x1 branch. Deployment uses /reparam per-group kernel+bias and no BN.
/// No fitting/fusion is hidden; DeployRepVGG explicitly fills deployed leaves.
///
/// @param p Mutable floating store, lazy training/deployed leaves.
/// @param state Mutable matching-dtype BN state; unused in deploy form.
/// @param b Valid block prefix/geometry and identity flag.
/// @param x Nonempty floating NHWC input with b.in channels.
/// @param deploy True uses one biased grouped 3x3 and requires training=false.
/// @param training Use batch BN when true in training form; false saved moments.
/// @param momentum BN retention in [0,1), default .9.
/// @param update_stats Permit eager training BN updates, default true.
/// @return NHWC ReLU after aligned branch sum, or equivalent deployed filter.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RepVGGBlockSpec b{"b", 8, 8, 1, 2, true};
/// const auto y = n::models::RepVGGBlock(p, state, b, n::ones({2, 5, 5, 8}), false, false);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline nn::Tensor RepVGGBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                              const RepVGGBlockSpec &b, const nn::Tensor &x, bool deploy,
                              bool training, double momentum = .9, bool update_stats = true) {
  b.Validate();
  if (x.dim() != 4 || x.numel() == 0 || x.size(3) != b.in || (deploy && training))
    throw std::invalid_argument("invalid RepVGG block input/mode");
  nn::Tensor y;
  if (deploy) {
    nn::ConvLayerOptions conv;
    conv.init = nn::init::he_uniform();
    conv.stride = b.stride;
    conv.padding = nn::Padding::Valid;
    y = GroupedConv2D(p, b.name + "/reparam", nn::pad(x, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}),
                      b.out, 3, b.groups, conv);
  } else {
    y = RepVGGBranch(p, state, b.name + "/dense", x, b.out, 3, b.stride, b.groups, training,
                     momentum, update_stats) +
        RepVGGBranch(p, state, b.name + "/pointwise", x, b.out, 1, b.stride, b.groups,
                     training, momentum, update_stats);
    if (b.identity)
      y = y + nn::batch_norm(p, state, b.name + "/identity/bn", x, training,
                             RepVGGNorm(momentum, update_stats));
  }
  return nn::relu(y);
}

/// @brief Preset, classifier, stem, deployment and BN metadata for RepVGG.
///
/// Value metadata, no learned state or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::RepVGGConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGG(), RepVGGBlock(), FuseRepVGGBlock()
struct RepVGGConfig {
  std::string model = "A0"; ///< One of ten released names from RepVGGLayout().
  int classes = 1000;       ///< At least two classifier outputs.
  bool small_stem =
      false;            ///< Change first block stride to one; all four stage strides stay two.
  bool deploy = false;  ///< Single biased 3x3 blocks; requires training=false.
  double momentum = .9; ///< BN moving retention in [0,1).
  bool update_stats = true; ///< Permit eager moving updates in training form.
};

/// @brief Build a complete released RepVGG training-form or deployed classifier.
///
/// Stem is a stride-two block (one for small_stem), followed by four
/// all-stride-two stage entries with released repeats/widths, global mean
/// and biased classifier. A0 has 22 blocks including stem. g2/g4 presets
/// group only even block indices 2..26 after stem. He-uniform filters and
/// BN eps 1e-5; no dropout, preprocessing, loss, optimizer or fitting.
/// Store mutations require synchronization; retained graphs borrow leaves.
///
/// @param p Mutable floating learned store, lazy named leaves.
/// @param state Mutable matching-dtype BN store, unused for deployed filters.
/// @param image Nonempty floating RGB NHWC images matching p dtype.
/// @param cfg Valid preset/classes/stem/deploy/BN metadata, default A0.
/// @param training True uses batch BN, false saved moments; deploy forbids true.
/// @return Final features, biased classifier logits, every block spec and post-ReLU map.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RepVGGConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::RepVGG(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline RepVGGOutputs RepVGG(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const RepVGGConfig &cfg = {},
                            bool training = false) {
  const auto layout = RepVGGLayout(cfg.model);
  if (image.dim() != 4 || image.numel() == 0 || image.size(3) != 3 ||
      !(cfg.momentum >= 0 && cfg.momentum < 1) || cfg.classes < 2 || (cfg.deploy && training))
    throw std::invalid_argument("invalid RepVGG configuration");
  RepVGGOutputs out;
  /// Execute and record one block map/specification in model order.
  auto block = [&](const RepVGGBlockSpec &b, const nn::Tensor &x) {
    out.blocks.push_back(b);
    out.outputs.push_back(
        RepVGGBlock(p, state, b, x, cfg.deploy, training, cfg.momentum, cfg.update_stats));
    return out.outputs.back();
  };
  int in = std::min(64, int(64 * layout.width[0]));
  nn::Tensor h = block({"stage0", 3, in, cfg.small_stem ? 1 : 2, 1, false}, image);
  int index = 1;
  for (int stage = 0; stage < 4; ++stage) {
    const int planes = int((64 << stage) * layout.width[stage]);
    for (int i = 0; i < layout.blocks[stage]; ++i, ++index) {
      const int stride = i ? 1 : 2, groups = layout.groups > 1 && index % 2 == 0 && index <= 26
                                                 ? layout.groups
                                                 : 1;
      h = block({"stage" + std::to_string(stage + 1) + "/block" + std::to_string(i + 1), in,
                 planes, stride, groups, in == planes && stride == 1},
                h);
      in = planes;
    }
  }
  out.features = h;
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

/// @brief Tensor-packed kernels and summed bias of one inference RepVGG block.
///
/// Tensor handles own storage references and retain any active differentiation
/// graphs. Parameter/state stores retain source learned leaves; no fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RepVGGBlockSpec b{"b", 8, 8, 1, 2, true};
/// n::models::RepVGGBlock(p, state, b, n::ones({1, 5, 5, 8}), false, false);
/// const auto fused = n::models::FuseRepVGGBlock(b, p, state);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGG(), RepVGGBlock(), FuseRepVGGBlock()
struct FusedBlock {
  std::vector<nn::Tensor> kernels; ///< One [3,3,Cin/G,Cout/G] tensor per group, source dtype.
  nn::Tensor
      bias; ///< Full [Cout] bias; preserves source affine/filter gradients when enabled.
};
/// @brief Fold fixed inference BN values and sum aligned branch kernels with tensors.
///
/// Each branch folds W*=gamma/sqrt(var+1e-5), b*=beta-mean*scale.
/// The 1x1 filter pads to a 3x3 centre; identity becomes a grouped centre
/// one-hot kernel with its own folded BN. Kernels and biases SUM; ReLU
/// remains outside. Uses fold_batch_norm, pad, one_hot and tensor addition,
/// never host Float32 weight vectors. When recording, retained source
/// filter/gamma/beta derivatives are preserved; moments are fixed for
/// inference equivalence. Does not reproduce batch-dependent training BN.
/// Work/storage O(9*Cin*Cout/G), no fitting or source mutation.
///
/// @param b Valid initialized training-form block specification.
/// @param p Borrowed floating learned store; source values never mutated.
/// @param state Borrowed matching-dtype saved BN mean/variance store.
/// @return Per-group fused 3x3 kernels and full bias, preserving Float32/64 dtype.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @throws std::runtime_error If an initialized filter or BN value is missing.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RepVGGBlockSpec b{"b", 8, 8, 1, 2, true};
/// n::models::RepVGGBlock(p, state, b, n::ones({1, 5, 5, 8}), false, false);
/// const auto fused = n::models::FuseRepVGGBlock(b, p, state);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline FusedBlock FuseRepVGGBlock(const RepVGGBlockSpec &b, const nn::ParameterStore &p,
                                  const nn::ParameterStore &state) {

  b.Validate();
  if (p.dtype() != state.dtype())
    throw std::invalid_argument("RepVGG learned/state dtypes differ");
  const int ci = b.in / b.groups, co = b.out / b.groups;
  /// Borrow a required tensor without casting through a host weight vector.
  auto get = [&](const nn::ParameterStore &store, const std::string &key) {
    if (!store.contains(key))
      throw std::runtime_error("missing RepVGG value " + key);
    return store.at(key);
  };
  /// Slice full-width BN vectors to one contiguous output group.
  auto fold = [&](const nn::Tensor &weight, const std::string &branch, int group) {
    const auto bn = b.name + "/" + branch + "/bn/";
    /// Borrow one contiguous group of a required full-width BN vector.
    auto part = [&](const nn::ParameterStore &store, const std::string &key) {
      return nn::slice_axis(get(store, bn + key), 0, group * co, (group + 1) * co);
    };
    return nn::fold_batch_norm(weight, nn::Tensor(), part(p, "gamma"), part(p, "beta"),
                               part(state, "moving_mean"), part(state, "moving_var"), 1e-5);
  };
  FusedBlock fused;
  std::vector<nn::Tensor> shifts;
  for (int group = 0; group < b.groups; ++group) {
    const auto suffix = b.groups > 1 ? "/group" + std::to_string(group) : "";
    const auto dense = fold(get(p, b.name + "/dense" + suffix + "/W"), "dense", group),
               point = fold(get(p, b.name + "/pointwise" + suffix + "/W"), "pointwise", group);
    auto kernel = dense.weight + nn::pad(point.weight, {{1, 1}, {1, 1}, {0, 0}, {0, 0}});
    auto bias = dense.bias + point.bias;
    if (b.identity) {
      const auto centre =
          nn::one_hot(nn::arange(0, ci).to(nn::DType::Int64), co, 1, 0, p.dtype())
              .reshape({1, 1, ci, co});
      const auto identity =
          fold(nn::pad(centre, {{1, 1}, {1, 1}, {0, 0}, {0, 0}}), "identity", group);
      kernel = kernel + identity.weight;
      bias = bias + identity.bias;
    }
    fused.kernels.push_back(kernel);
    shifts.push_back(bias);
  }
  fused.bias = nn::concat(shifts, 0);
  return fused;
}

/// @brief Copy fused filters/biases and classifier into an inference deployment store.
///
/// Fuses each block under NoGradGuard, detaches deployment leaves and
/// copies optional output/W and output/b with independent storage.
/// Source moving moments must match the selected parameter checkpoint.
/// Set cfg.deploy=true and training=false for the subsequent forward.
/// No optimizer or deployment forward runs here; no BN state is copied.
/// Existing incompatible destination shapes can fail after earlier writes,
/// so use a fresh destination for transactional conversion. Synchronize stores.
///
/// @param deployed Mutable independent learned store matching p dtype; creates/replaces named leaves.
/// @param blocks Initialized training-form specs, normally RepVGGOutputs.blocks.
/// @param p Borrowed source learned store; not modified.
/// @param state Borrowed fixed inference BN moments matching p dtype.
/// @throws std::invalid_argument For invalid geometry/settings/dtype or incompatible stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RepVGGConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::RepVGG(p, state, n::ones({1, 32, 32, 3}), cfg);
/// n::ParameterStore deployed(19);
/// n::models::DeployRepVGG(deployed, y.blocks, p, state);
/// @endcode
/// @par References
/// - Ding et al. (2021), https://arxiv.org/abs/2101.03697; https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
/// @see RepVGGBlockSpec, RepVGG(), FuseRepVGGBlock(), DeployRepVGG()
inline void DeployRepVGG(nn::ParameterStore &deployed,
                         const std::vector<RepVGGBlockSpec> &blocks,
                         const nn::ParameterStore &p, const nn::ParameterStore &state) {
  if (deployed.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("RepVGG deployment store dtypes differ");
  nn::NoGradGuard guard;
  for (const auto &b : blocks) {
    const auto fused = FuseRepVGGBlock(b, p, state);
    const int co = b.out / b.groups;
    for (int g = 0; g < b.groups; ++g) {
      const auto prefix =
          b.name + "/reparam" + (b.groups > 1 ? "/group" + std::to_string(g) : "");
      deployed.set(prefix + "/W", fused.kernels[g]);
      deployed.set(prefix + "/b", nn::slice(fused.bias, {g * co}, {co}));
    }
  }
  for (const auto &name : {"output/W", "output/b"})
    if (p.contains(name))
      deployed.set(name, p.at(name).clone());
}
} // namespace datamunge::nn::models
