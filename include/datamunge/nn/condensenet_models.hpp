#pragma once
/// @file condensenet_models.hpp
/// @brief Tensor-ranked condensation, packed inference and complete dense CondenseNet classifiers.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/classic_cnn_components.hpp"
#include "datamunge/nn/mobilenet_models.hpp"
#include "datamunge/nn/model_plot.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;

/// @brief Borrowed stores, condensing progress and optional penalty/output logs.
///
/// Referenced stores/logs must outlive each synchronous call; copies retain
/// references, not independent state. Callers synchronize mutations. Progress and
/// penalty coefficient are metadata; weight importance/mask updates use tensors.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see LearnedGroupConv()
struct CondenseContext {
  nn::ParameterStore& p; ///< Borrowed floating learned leaves, lazily created/reused.
  nn::ParameterStore&
      state; ///< Borrowed same-dtype BN moments, masks/stages and optional run step.
  std::vector<MobileLayer>& layers;   ///< Appended dense-mask/group-filter arithmetic metadata.
  bool                      training; ///< Use batch moments, optional lasso and condensing updates.
  double                    momentum = .9; ///< BN moving retention in [0,1).
  double                    progress = 0;  ///< Finite completed-step fraction in [0,1].
  std::vector<nn::Tensor>*  penalties =
      nullptr;      ///< Optional borrowed list, appended scaled group-lasso terms.
  double lasso = 0; ///< Finite nonnegative penalty coefficient; zero disables penalty collection.
  bool   update_stats = true; ///< Permit moving BN/mask/stage updates in training mode.
};
/// @brief Normalized features, masked convolution output and the exact mask used by this call.
///
/// Handles retain storage/graphs. mask is a detached cloned snapshot taken
/// BEFORE the call updates stored condensing state, so backward remains valid.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// const auto y = n::models::LearnedGroupConv(c, "lgc", n::ones({2, 5, 5, 8}), 8, 2, 2);
/// @endcode
/// @see LearnedGroupConv()
struct LearnedGroupConvOutputs {
  nn::Tensor normalized; ///< BN/ReLU input map, NHWC.
  nn::Tensor output; ///< Bias-free masked 1x1 convolution output retaining weight/input gradients.
  nn::Tensor mask;   ///< Detached [Cin,Cout] mask snapshot actually used, before schedule update.
};
/// @brief Apply learned-scale CondenseNet BN with epsilon 1e-5, followed by ReLU.
///
/// Moving moments update only in training with update_stats=true.
/// No fitting; O(numel(x)) work/storage.
///
/// @param c Borrowed learned/state stores, BN mode and update setting.
/// @param name BN prefix.
/// @param x Nonempty floating NHWC features matching store dtype.
/// @return Same-shaped BN/ReLU map retaining gradients.
/// @throws std::invalid_argument For invalid dimensions, settings, masks/stages, dtype,
/// normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// const auto y = n::models::CondenseBNReLU(c, "bn", n::ones({2, 5, 5, 8}));
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet(), LearnedGroupConv(), Condense()
inline nn::Tensor CondenseBNReLU(CondenseContext& c, const std::string& name, const nn::Tensor& x) {
  nn::BatchNormOptions bn;
  bn.momentum     = c.momentum;
  bn.eps          = 1e-5;
  bn.update_stats = c.update_stats;
  return nn::relu(nn::batch_norm(c.p, c.state, name, x, c.training, bn));
}

/// @brief Prune lowest-importance input channels per output group using tensor ranks.
///
/// Target stage is min(C-1,floor(float(progress)*2*(C-1))). At each
/// new stage drop Cin/C still-active input channels with smallest summed
/// absolute masked weights across that group outputs. Stable tensor argsort
/// breaks equal scores by lower input index. Skipped stages are caught up,
/// unlike a one-event update; decreasing progress never resurrects channels.
/// Writes new detached masks/stage in state, never p. Finite weights, binary
/// group-consistent masks and stage counts are validated. Work O(G*Cin log Cin)
/// per new stage plus weight reduction, no gradients through pruning decisions.
/// Missing initialized store keys throw std::out_of_range.
///
/// @param p Borrowed learned store with initialized name/W.
/// @param state Borrowed store with initialized binary group-consistent name/mask and integer
/// name/stage.
/// @param name Learned group convolution prefix.
/// @param groups Positive interleaved output-group count, divides Cout.
/// @param condense Factor C>=2 dividing Cin; final retained count Cin/C.
/// @param progress Finite completed-step fraction in [0,1].
/// @throws std::invalid_argument For invalid dimensions, settings, masks/stages, dtype,
/// normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// const auto y = n::models::LearnedGroupConv(c, "lgc", n::ones({2, 5, 5, 8}), 8, 2, 2);
/// n::models::UpdateCondensation(p, state, "lgc", 2, 2, .5);
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet(), LearnedGroupConv(), Condense()
inline void UpdateCondensation(
    nn::ParameterStore& p,
    nn::ParameterStore& state,
    const std::string&  name,
    int                 groups,
    int                 condense,
    double              progress) {
  if (groups < 1 || condense < 2 || !(progress >= 0 && progress <= 1))
    throw std::invalid_argument("invalid condensation groups, ratio or progress");
  nn::NoGradGuard guard;
  auto            mask = state.at(name + "/mask");
  if (mask.dim() != 2 || mask.numel() == 0 || mask.size(0) % condense || mask.size(1) % groups)
    throw std::invalid_argument("invalid condensation mask dimensions");
  const auto in = mask.size(0), out = mask.size(1), per = out / groups;
  const auto weight = p.at(name + "/W");
  if (weight.numel() != in * out || weight.dtype() != mask.dtype() ||
      !nn::all(nn::less(nn::abs(weight), std::numeric_limits<double>::infinity())).item())
    throw std::invalid_argument("condensation needs matching finite weights");
  const double old = state.at(name + "/stage").item();
  if (!std::isfinite(old) || old < 0 || old > condense - 1 || std::floor(old) != old)
    throw std::invalid_argument("invalid condensation stage");
  const auto view = mask.reshape({in, per, groups}), active = nn::slice_axis(view, 1, 0, 1);
  if (!nn::all(nn::equal(mask * (mask - 1), 0)).item() || !nn::all(nn::equal(view, active)).item())
    throw std::invalid_argument("mask must be binary and group-consistent");
  if (!nn::all(nn::equal(nn::sum(active, {0, 1}), double(in - int(old) * (in / condense)))).item())
    throw std::invalid_argument("mask count does not match condensation stage");
  const int target =
      std::min(condense - 1, int(std::floor(float(progress) * 2.f * float(condense - 1))));
  for (int stage = int(old); stage < target; ++stage) {
    const auto current = mask.reshape({in, per, groups});
    const auto live    = nn::transpose(nn::squeeze(nn::slice_axis(current, 1, 0, 1), {1}), {1, 0});
    const auto importance = nn::transpose(
        nn::sum(nn::abs(weight.reshape({in, out}) * mask).reshape({in, per, groups}), {1}), {1, 0});
    const auto scores = nn::where(
        nn::greater(live, .5),
        importance,
        nn::full(importance.shape(), std::numeric_limits<double>::infinity(), importance.dtype()));
    const auto ranks    = nn::argsort(nn::argsort(scores));
    const auto retained = live * nn::greater_equal(ranks, in / condense).to(mask.dtype());
    mask =
        (nn::expand_dims(nn::transpose(retained, {1, 0}), 1) * nn::ones({1, per, 1}, mask.dtype()))
            .reshape({in, out});
  }
  if (target > old) {
    state.set(name + "/mask", mask);
    state.set(name + "/stage", nn::scalar(target, mask.dtype()));
  }
}
/// @brief Apply BN/ReLU and a tensor-masked 1x1 learned group convolution.
///
/// Lazily creates dense W, ones mask and stage zero. Output uses the current
/// mask snapshot. Training optionally appends scaled lasso=sum sqrt(max(
/// sum_group(masked_W²),1e-6)) while stage<C-1; callers add it to loss.
/// Then update_stats=true prunes state from pre-optimizer weights. New masks
/// apply on the NEXT forward. Dense kernels are still computed while masked;
/// training does not gain packed grouped-convolution speed. No optimizer hidden.
///
/// @param c Borrowed stores/logs and explicit training/progress/lasso settings.
/// @param name Prefix for bn, W, mask and stage leaves.
/// @param x Nonempty floating NHWC input matching stores.
/// @param out Positive output width divisible by groups.
/// @param groups Positive interleaved output-group count dividing input and output.
/// @param condense Factor >=2 dividing input width.
/// @return Normalized input, output and detached pre-update mask snapshot with graphs.
/// @throws std::invalid_argument For invalid dimensions, settings, masks/stages, dtype,
/// normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// const auto y = n::models::LearnedGroupConv(c, "lgc", n::ones({2, 5, 5, 8}), 8, 2, 2);
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet(), LearnedGroupConv(), Condense()
inline LearnedGroupConvOutputs LearnedGroupConv(
    CondenseContext&   c,
    const std::string& name,
    const nn::Tensor&  x,
    std::int64_t       out,
    int                groups,
    int                condense) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || groups < 1 || condense < 2 ||
      !(c.progress >= 0 && c.progress <= 1) || !std::isfinite(c.lasso) || c.lasso < 0)
    throw std::invalid_argument("invalid learned group convolution settings");
  const std::int64_t in = x.shape()[3];
  if (in % groups || out % groups || in % condense || condense < 2)
    throw std::invalid_argument("invalid learned group convolution");
  LearnedGroupConvOutputs r;
  r.normalized            = CondenseBNReLU(c, name + "/bn", x);
  const nn::Tensor weight = c.p.get(name + "/W", {1, 1, in, out}, nn::init::he_uniform());
  // A detached copy: the condensing update below rewrites the stored mask in place.
  r.mask = c.state.get(name + "/mask", {in, out}, nn::init::ones()).detach().clone();
  c.state.get(name + "/stage", {}, nn::init::zeros());
  const nn::Tensor masked = nn::reshape(weight, {in, out}) * r.mask;
  r.output                = nn::conv2d(
      r.normalized, nn::reshape(masked, {1, 1, in, out}), nn::conv_options(1, nn::Padding::Valid));
  c.layers.push_back({name, r.output.shape(), in * out});
  if (!c.training)
    return r;
  if (c.penalties && c.lasso > 0 && c.state.at(name + "/stage").item<float>() < condense - 1) {
    const nn::Tensor squared =
        nn::sum(nn::reshape(nn::square(masked), {in, out / groups, groups}), {1});
    const nn::Tensor lasso =
        nn::sum(nn::sqrt(nn::maximum(squared, nn::scalar(1e-6, masked.dtype()))));
    c.penalties->push_back(lasso * c.lasso);
  }
  if (c.update_stats)
    UpdateCondensation(c.p, c.state, name, groups, condense, c.progress);
  return r;
}

/// @brief Dense-stage dimensions, group condensation and regularization settings.
///
/// Defaults describe CIFAR CondenseNet-86: 42 dense layers, growth 8/16/32.
/// Metadata validation allocates no learned tensors; no fitting is hidden.
///
/// @par Example
/// @code
/// dm::nn::models::CondenseNetConfig cfg;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet()
struct CondenseNetConfig {
  std::vector<int> stages = {14, 14, 14}; ///< Positive layer counts per dense stage.
  std::vector<int> growth = {8, 16, 32};  ///< New channels per layer in each stage.
  int              groups = 4;            ///< Interleaved output groups for learned 1x1 filters.
  int    condense   = 4;     ///< Final per-group retained input fraction is 1/condense, factor >=2.
  int    bottleneck = 4;     ///< LGC output width is this times stage growth.
  int    classes    = 10;    ///< At least two classifier outputs.
  bool   imagenet   = false; ///< Stem stride two rather than one.
  double lasso      = 1e-5;  ///< Nonnegative coefficient for pre-final-stage group-lasso penalties.
  double momentum   = .9;    ///< BN moving retention in [0,1).
  bool   update_stats = true; ///< Allow training BN/masks/stages/run-step mutations.

  /// @brief Validate stage lengths, divisibility, probabilities and representable channel widths.
  ///
  /// Every layer input must divide by groups and condense; every growth and LGC
  /// output must divide by groups. O(number of dense layers), no learned allocation.
  ///
  /// @throws std::invalid_argument For invalid stages, widths, overflow, classes, lasso or
  /// momentum.
  /// @par Example
  /// @code
  /// dm::nn::models::CondenseNetConfig{}.Validate();
  /// @endcode
  /// @see CondenseNetConfig
  void Validate() const {
    if (stages.empty() || stages.size() != growth.size() || groups < 1 || condense < 2 ||
        bottleneck < 1 || classes < 2 || !std::isfinite(lasso) || lasso < 0 ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid CondenseNet configuration");
    int64_t channels = 2LL * growth[0];
    for (size_t s = 0; s < stages.size(); ++s) {
      if (stages[s] < 1 || growth[s] < 1 || growth[s] % groups ||
          (int64_t(bottleneck) * growth[s]) % groups ||
          int64_t(bottleneck) * growth[s] > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid CondenseNet stage widths");
      for (int layer = 0; layer < stages[s]; ++layer) {
        if (channels < 1 || channels % groups || channels % condense ||
            channels > std::numeric_limits<int>::max() - growth[s])
          throw std::invalid_argument("invalid CondenseNet learned group widths");
        channels += growth[s];
      }
    }
  }
};
/// @brief Final features/logits, arithmetic metadata and explicit group-lasso penalty terms.
///
/// Tensor handles retain storage/graphs. Caller must add penalties to its loss
/// explicitly and own the optimizer loop. State remains in the borrowed store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::CondenseNetConfig cfg;
/// cfg.stages = {1, 1, 1};
/// const auto y = n::models::CondenseNet(p, state, n::ones({2, 8, 8, 3}), cfg);
/// @endcode
/// @see CondenseNet()
struct CondenseNetOutputs {
  nn::Tensor features; ///< Final BN/ReLU spatial map before global mean.
  nn::Tensor logits;   ///< Unnormalised [N,classes] dense scores.
  std::vector<MobileLayer>
      layers; ///< Training/masked dense 1x1 and grouped 3x3 arithmetic metadata.
  std::vector<nn::Tensor> penalties; ///< Scaled group-lasso terms; add explicitly to the objective.
  int                     channels = 0; ///< Final feature-map width before classifier.
};
/// @brief Build the complete dense CondenseNet tensor classifier without fitting.
///
/// Architecture tracing groups each dense stage, including its following transition.
/// The first stage includes the stem; layer/operation views remain expandable.
///
/// Stem is bias-free 3x3 to twice first growth; each layer appends growth
/// channels from LGC to bottleneck*growth then BN/ReLU/grouped 3x3.
/// Stages have only 2x2 average-pool transitions, no compression convolution.
/// Final BN/ReLU, global mean and biased classifier. No dropout. He-uniform
/// filters, BN eps 1e-5. One enabled training forward advances run step
/// after all layers; progress before that increment drives masks. Warm all
/// leaves in inference before constructing an optimizer. Inference never
/// advances moments/masks/step. Caller must explicitly add returned penalties,
/// choose loss/optimizer, and own minibatches/checkpoints. Save state alongside
/// weights; a selected early checkpoint may not yet be fully condensed.
///
/// @param p Mutable learned store, lazily creates/reuses named leaves.
/// @param state Mutable BN/mask/stage store, plus condense/step on training calls.
/// @param image Nonempty floating NHWC images matching stores.
/// @param cfg Valid dense stages, growth and condensation metadata.
/// @param training True uses batch moments, collects lasso and optionally advances condensation.
/// @param total_steps Positive total training-forward budget when training; ignored for inference.
/// @return Features/logits, explicit penalty tensors and dense-mask arithmetic metadata.
/// @throws std::invalid_argument For invalid dimensions, settings, masks/stages, dtype,
/// normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::CondenseNetConfig cfg;
/// cfg.stages = {1, 1, 1};
/// const auto y = n::models::CondenseNet(p, state, n::ones({2, 8, 8, 3}), cfg);
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet(), LearnedGroupConv(), Condense()
inline CondenseNetOutputs CondenseNet(
    nn::ParameterStore&      p,
    nn::ParameterStore&      state,
    const nn::Tensor&        image,
    const CondenseNetConfig& cfg         = {},
    bool                     training    = false,
    std::int64_t             total_steps = 0) {
  cfg.Validate();
  if (image.dim() != 4 || image.numel() == 0 || (training && total_steps < 1))
    throw std::invalid_argument("invalid CondenseNet configuration");
  CondenseNetOutputs out;
  CondenseContext    c{
      p, state, out.layers, training, cfg.momentum, 0, &out.penalties, cfg.lasso, cfg.update_stats};
  if (training) {
    // One training step per training-mode pass; progress is the fraction completed.
    const nn::Tensor step = state.get("condense/step", {}, nn::init::zeros());
    c.progress            = double(step.item<float>()) / double(total_steps);
  }
  int                  channels = 2 * cfg.growth[0];
  nn::ConvLayerOptions stem;
  stem.bias    = false;
  stem.init    = nn::init::he_uniform();
  stem.stride  = cfg.imagenet ? 2 : 1;
  stem.padding = nn::Padding::Valid;
  nn::Tensor h =
      nn::conv2d(p, "stem", nn::pad(image, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), channels, 3, stem);
  c.layers.push_back({"stem", h.shape(), 27LL * channels});
  nn::Tensor stage_input = nn::model_plot_detail::recording() ? image : nn::Tensor{};
  for (std::size_t stage = 0; stage < cfg.stages.size(); ++stage) {
    const int k = cfg.growth[stage], wide = cfg.bottleneck * k;
    for (int layer = 0; layer < cfg.stages[stage]; ++layer) {
      const auto name = "stage" + std::to_string(stage + 1) + "/layer" + std::to_string(layer + 1);
      nn::Tensor y = LearnedGroupConv(c, name + "/lgc", h, wide, cfg.groups, cfg.condense).output;
      nn::ConvLayerOptions conv;
      conv.bias    = false;
      conv.init    = nn::init::he_uniform();
      conv.padding = nn::Padding::Valid;
      y            = GroupedConv2D(
          p,
          name + "/conv",
          nn::pad(CondenseBNReLU(c, name + "/conv/bn", y), {{0, 0}, {1, 1}, {1, 1}, {0, 0}}),
          k,
          3,
          cfg.groups,
          conv);
      c.layers.push_back({name + "/conv", y.shape(), 9LL * wide * k / cfg.groups});
      h = nn::concat({h, y}, 3);
      channels += k;
    }
    if (stage + 1 < cfg.stages.size())
      h = nn::avg_pool2d(h, 2, 2, nn::Padding::Valid);
    if (nn::model_plot_detail::recording()) {
      nn::model_stage(
          "Dense stage " + std::to_string(stage + 1) + " / " + std::to_string(cfg.stages[stage]) +
              " layers",
          stage_input,
          h);
      stage_input = h;
    }
  }
  out.features = CondenseBNReLU(c, "final/bn", h);
  out.logits   = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  out.channels = channels;
  if (training && cfg.update_stats) {
    nn::NoGradGuard guard;
    state.set("condense/step", state.at("condense/step") + 1.0);
  }
  return out;
}
/// @brief Gather indices and packed grouped kernel for a converted learned group convolution.
///
/// Index metadata lists group-surviving inputs in increasing order, group by
/// group; duplicates across groups are permitted. Weight packing uses tensor
/// gather/concat and preserves dtype/weight gradients when recording is enabled.
/// BN affine parameters and moving moments must be gathered with the same index.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// const auto y = n::models::LearnedGroupConv(c, "lgc", n::ones({2, 5, 5, 8}), 8, 2, 2);
/// const auto packed = n::models::Condense(p.at("lgc/W"), state.at("lgc/mask"), 2);
/// @endcode
/// @see Condense(), PaperChannelShuffle()
struct CondensedConv {
  std::vector<std::int64_t>
      index; ///< Group-ordered retained input IDs, possibly repeated across groups.
  nn::Tensor
      weight; ///< [1,1,retained_per_group,Cout], contiguous group output columns before shuffle.
};

/// @brief Pack a binary-masked learned kernel with tensor gathers, preserving dtype.
///
/// Output o belongs to o mod G in the masked kernel. Packed outputs are
/// contiguous per group, requiring PaperChannelShuffle afterward. Gather
/// inputs, BN affine/moving values with index, normalize/ReLU, then grouped
/// conv2d with packed weights. Source weights are not mutated or cast through
/// host arrays. Discrete index selection is nondifferentiable; retained weight
/// gathers preserve gradients when enabled. Does not halve the classifier.
///
/// @param weight Floating Float32/64 tensor with Cin*Cout values, any original shape.
/// @param mask Binary group-consistent [Cin,Cout] tensor.
/// @param groups Positive divisor of Cout; every group must retain the same positive input count.
/// @return Group-ordered input IDs and packed [1,1,width,Cout] weight tensor.
/// @throws std::invalid_argument For invalid dimensions, settings, masks/stages, dtype,
/// normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::CondenseContext c{p, state, layers, false};
/// const auto y = n::models::LearnedGroupConv(c, "lgc", n::ones({2, 5, 5, 8}), 8, 2, 2);
/// const auto packed = n::models::Condense(p.at("lgc/W"), state.at("lgc/mask"), 2);
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet(), LearnedGroupConv(), Condense()
inline CondensedConv Condense(const nn::Tensor& weight, const nn::Tensor& mask, int groups) {
  if (mask.dim() != 2 || mask.numel() == 0 || groups < 1 || mask.size(1) % groups ||
      weight.numel() != mask.numel() ||
      (weight.dtype() != nn::DType::Float32 && weight.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid condensed convolution dimensions/dtype");
  const auto in = mask.size(0), out = mask.size(1), per = out / groups;
  const auto typed_mask = mask.to(weight.dtype());
  const auto view       = typed_mask.reshape({in, per, groups}),
             active     = nn::squeeze(nn::slice_axis(view, 1, 0, 1), {1});
  if (!nn::all(nn::equal(typed_mask * (typed_mask - 1), 0)).item() ||
      !nn::all(nn::equal(view, nn::expand_dims(active, 1))).item())
    throw std::invalid_argument("conversion needs binary group-consistent masks");
  CondensedConv           r;
  std::vector<nn::Tensor> kernels;
  int64_t                 width = -1;
  for (int group = 0; group < groups; ++group) {
    const auto live  = nn::slice_axis(active, 1, group, group + 1).reshape({in});
    const auto count = nn::count_nonzero(live).item<int64_t>();
    if (count < 1 || (width >= 0 && width != count))
      throw std::invalid_argument("conversion needs positive equal retained widths");
    width               = count;
    const auto index    = nn::slice_axis(nn::argsort(1 - live), 0, 0, width);
    const auto metadata = index.to_vector<int64_t>();
    r.index.insert(r.index.end(), metadata.begin(), metadata.end());
    kernels.push_back(
        nn::gather(nn::slice_axis(weight.reshape({in, out}), 1, group, out, groups), index, 0));
  }
  r.weight = nn::concat(kernels, 1).reshape({1, 1, width, out});
  return r;
}

/// @brief Analytical RGB inference-form count metadata, without allocating a converted model.
///
/// Assumes final condensation and optionally a classifier reading half the final
/// features, as in the release. Does not perform mask conversion or classifier
/// feature selection, measure speed or count BN/pooling/activation arithmetic.
///
/// @par Example
/// @code
/// const auto c = dm::nn::models::ConvertedCondenseNetCost({}, 32); // 516202 parameters
/// @endcode
/// @see ConvertedCondenseNetCost()
struct CondenseNetConvertedCost {
  int64_t parameters; ///< Learned kernel, bias and duplicated BN affine scalars in the assumed
                      ///< converted layout.
  int64_t macs; ///< Kernel/classifier MACs per RGB image, excluding BN/activation/pooling/gather.
  int     features;            ///< Full final dense feature width.
  int     classifier_features; ///< Classifier input width, full or integer half as requested.
};
/// @brief Estimate a fully condensed RGB model budget from architecture metadata.
///
/// Counts gathered/duplicated BN affine values, packed LGC weights, grouped
/// 3x3, final BN and classifier. Inference conversion itself is not performed;
/// classifier half-selection is an explicit ASSUMPTION. Stem assumes RGB,
/// stride two for ImageNet. Work O(total layers); no model allocation/fitting.
///
/// @param cfg Valid dense/growth/condensation metadata.
/// @param resolution Positive square RGB input side, sufficient for pooling.
/// @param halve_classifier True assumes a classifier using half the final features, default true.
/// @return Analytical learned parameter count, kernel/classifier MACs and feature widths.
/// @throws std::invalid_argument For invalid dimensions, settings, masks/stages, dtype,
/// normalization or stored shapes.
/// @par Example
/// @code
/// const auto c = dm::nn::models::ConvertedCondenseNetCost({}, 32); // {516202,62377888,800,400}
/// @endcode
/// @par References
/// - Huang et al. (2018), https://arxiv.org/abs/1711.09224;
/// https://github.com/ShichenLiu/CondenseNet
/// @see CondenseNet(), LearnedGroupConv(), Condense()
inline CondenseNetConvertedCost ConvertedCondenseNetCost(
    const CondenseNetConfig& cfg,
    int                      resolution,
    bool                     halve_classifier = true) {
  cfg.Validate();
  if (resolution < 1)
    throw std::invalid_argument("positive converted resolution required");
  int64_t params = 0, macs = 0;
  int     channels = 2 * cfg.growth[0], side = cfg.imagenet ? (resolution + 1) / 2 : resolution;
  params += 27 * channels;
  macs += int64_t(side) * side * 27 * channels;
  for (size_t stage = 0; stage < cfg.stages.size(); ++stage) {
    const int k = cfg.growth[stage], wide = cfg.bottleneck * k;
    for (int layer = 0; layer < cfg.stages[stage]; ++layer, channels += k) {
      const int64_t gathered = int64_t(channels) * cfg.groups / cfg.condense;
      params += 2 * gathered + int64_t(gathered / cfg.groups) * wide + 2 * wide +
                int64_t(wide / cfg.groups) * k * 9;
      macs += int64_t(side) * side *
              (int64_t(gathered / cfg.groups) * wide + int64_t(wide / cfg.groups) * k * 9);
    }
    if (stage + 1 < cfg.stages.size()) {
      side /= 2;
      if (side < 1)
        throw std::invalid_argument("converted pooling geometry too small");
    }
  }
  const int head = halve_classifier ? channels / 2 : channels;
  params += 2 * channels + int64_t(head) * cfg.classes + cfg.classes;
  macs += int64_t(head) * cfg.classes;
  return {params, macs, channels, head};
}

} // namespace datamunge::nn::models
