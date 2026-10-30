#pragma once
/// @file detection_components.hpp
/// @brief Tensor SSD matching, region proposal heads/losses and patch classification; no fitting.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/boxes.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Assign distinct anchors to targets before applying an SSD overlap threshold.
///
/// Repeatedly selects the highest still-available anchor/target pair, breaking ties
/// by lower anchor then lower target index, even at zero IoU. Afterwards unmatched
/// anchors take their highest-IoU target at or above threshold, with later target
/// indices winning ties to preserve the source convention. Tensor ops perform all
/// overlap reductions and masking; scalar extraction only controls discrete indices.
/// No gradients, mutation or optimizer. O(A*G*G) work and O(A*G) storage.
///
/// @param overlaps Float32/64 [A,G] IoUs, finite and in [0,1]; G must not exceed A.
/// @param threshold Finite remaining-anchor positive cutoff in [0,1], default .5.
/// @return Detached Int64 [A]: target index or -1 for background; empty targets produce all -1.
/// @throws std::invalid_argument For rank/dtype, invalid IoUs/threshold or more targets than anchors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto match = n::models::MatchSsd(n::tensor({1., 1., .5, .5}, {2, 2})); // [0,1]
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325
/// @see BuildRpn(), nn::match_boxes()
inline nn::Tensor MatchSsd(const nn::Tensor &overlaps, double threshold = .5) {
  if (overlaps.dim() != 2 ||
      (overlaps.dtype() != nn::DType::Float32 && overlaps.dtype() != nn::DType::Float64) ||
      !std::isfinite(threshold) || threshold < 0 || threshold > 1)
    throw std::invalid_argument("SSD matching requires floating [A,G] and threshold in [0,1]");
  const auto A = overlaps.shape()[0], G = overlaps.shape()[1];
  if (G > A || (overlaps.numel() && !nn::all(nn::logical_and(nn::greater_equal(overlaps, 0.),
                                                             nn::less_equal(overlaps, 1.)))
                                         .item<bool>()))
    throw std::invalid_argument("invalid SSD IoUs or too few anchors");
  nn::NoGradGuard guard;
  auto match = nn::full({A}, -1., nn::DType::Int64);
  if (!G)
    return match;
  const auto ai = nn::arange(0, A).to(nn::DType::Int64),
             gi = nn::arange(0, G).to(nn::DType::Int64);
  auto eligible = nn::ones({A, G}, nn::DType::Bool);
  for (std::int64_t i = 0; i < G; ++i) {
    const auto chosen =
        nn::argmax(nn::where(eligible, overlaps, nn::full({}, -1., overlaps.dtype()))
                       .reshape({A * G}),
                   0)
            .item<std::int64_t>();
    const auto a = chosen / G, g = chosen % G;
    match =
        nn::where(nn::equal(ai, double(a)), nn::full({}, double(g), nn::DType::Int64), match);
    eligible = nn::logical_and(eligible,
                               nn::logical_and(nn::not_equal(ai.reshape({A, 1}), double(a)),
                                               nn::not_equal(gi.reshape({1, G}), double(g))));
  }
  const auto best = nn::max(overlaps, {1});
  const auto last = G - 1 - nn::argmax(nn::reverse(overlaps, {1}), 1);
  return nn::where(nn::logical_and(nn::less(match, 0.), nn::greater_equal(best, threshold)),
                   last, match);
}
/// @brief Evaluate elementwise Smooth-L1 with unit transition.
///
/// No input mutation, parameter creation or fitting. O(numel) work/storage.
///
/// @param difference Float32/64 tensor of any shape; NaNs propagate.
/// @return Same-shaped differentiable tensor: .5*d² for abs(d)<1, abs(d)-.5 otherwise.
/// @throws std::invalid_argument For nonfloating dtype.
/// @par Example
/// @code
/// const auto y = dm::nn::models::DetectorSmoothL1(dm::nn::tensor({-2., 0., .5}));
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see RpnLoss()
inline nn::Tensor DetectorSmoothL1(const nn::Tensor &difference) {
  if (difference.dtype() != nn::DType::Float32 && difference.dtype() != nn::DType::Float64)
    throw std::invalid_argument("SmoothL1 requires floating input");
  const auto a = nn::abs(difference);
  return nn::where(nn::less(a, 1.), nn::square(a) * .5, a - .5);
}
/// @brief Describe one square feature level sharing a region proposal head.
///
/// Architecture metadata and an owning tensor handle; no learned state or fitting.
/// All levels must share batch size, channels, anchors per cell and dtype.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::RpnLevel level{n::ones({1, 4, 4, 8}), 8, 4, 3};
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see BuildRpn()
struct RpnLevel {
  nn::Tensor features;  ///< Floating [N,cells,cells,channels] features; graph retained.
  int channels;         ///< Positive input width shared by every level.
  int cells;            ///< Positive square feature side for this level.
  int anchors_per_cell; ///< Positive anchor count shared by every level.
};
/// @brief Retain region proposal objectness and box-delta graphs.
///
/// Concatenation order is level, row, column, anchor. Delta coordinates are
/// (dy,dx,dh,dw). Tensor handles own storage; learned leaves belong to the store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto r = n::models::BuildRpn(p, {{n::ones({1, 2, 2, 4}), 4, 2, 3}}, 8);
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see BuildRpn(), RpnLoss()
struct Rpn {
  nn::Tensor objectness_logits; ///< Floating [N,A] linear objectness scores.
  nn::Tensor objectness;        ///< Floating [N,A] logistic objectness probabilities.
  nn::Tensor deltas;            ///< Floating [N,A,4] normalized box offsets.
  int anchors = 0;              ///< Total anchors across levels, checked to fit signed int.
};
/// @brief Apply one shared 3x3 ReLU RPN head to one or more feature levels.
///
/// SAME biased He convolutions preserve each level side. No batch normalization,
/// state updates, anchor decoding, matching or fitting. Output-level concatenation
/// must match the caller anchor ordering. Store and learned leaf values must remain
/// unchanged until backward. Complexity is ordinary shared convolution per level.
///
/// @param p Mutable Float32/64 learned store; lazily creates rpn/conv, rpn/objectness and rpn/deltas leaves.
/// @param levels Nonempty square feature levels with shared batch, channels, anchors per cell and store dtype.
/// @param hidden Positive shared convolution output width, default 128.
/// @return Objectness logits/probabilities, box deltas and total anchor count with gradients.
/// @throws std::invalid_argument For nonpositive dimensions, inconsistent levels/dtype, anchor-count overflow or incompatible existing store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto r = n::models::BuildRpn(p, {{n::ones({1, 2, 2, 4}), 4, 2, 3}}, 8);
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see RpnLoss()
inline Rpn BuildRpn(nn::ParameterStore &p, const std::vector<RpnLevel> &levels,
                    int hidden = 128) {
  if (levels.empty() || hidden < 1)
    throw std::invalid_argument("positive nonempty RPN configuration");
  Rpn r;
  std::vector<nn::Tensor> logits, deltas;
  nn::ConvLayerOptions relu, linear;
  relu.init = nn::init::he_uniform();
  relu.activation = nn::Activation::Relu;
  linear.init = nn::init::he_uniform();
  std::int64_t batch = -1;
  for (const auto &level : levels) {
    if (level.channels < 1 || level.cells < 1 || level.anchors_per_cell < 1 ||
        level.anchors_per_cell > std::numeric_limits<int>::max() / 4 ||
        level.features.dim() != 4 || level.features.shape()[0] < 1 ||
        level.features.shape()[1] != level.cells || level.features.shape()[2] != level.cells ||
        level.features.shape()[3] != level.channels || level.features.dtype() != p.dtype() ||
        (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) ||
        level.channels != levels[0].channels ||
        level.anchors_per_cell != levels[0].anchors_per_cell ||
        (batch != -1 && level.features.shape()[0] != batch))
      throw std::invalid_argument("RPN levels must share valid shape, dtype and head");
    batch = level.features.shape()[0];
    const auto cell = std::int64_t(level.cells);
    if (cell > std::numeric_limits<int>::max() / cell / level.anchors_per_cell)
      throw std::invalid_argument("RPN anchor count overflow");
    const auto count = cell * cell * level.anchors_per_cell;
    if (count > std::numeric_limits<int>::max() - r.anchors)
      throw std::invalid_argument("RPN anchor total overflow");
    r.anchors += int(count);
    const auto h = nn::conv2d(p, "rpn/conv", level.features, hidden, 3, relu);
    logits.push_back(
        nn::reshape(nn::conv2d(p, "rpn/objectness", h, level.anchors_per_cell, 1, linear),
                    {batch, count}));
    deltas.push_back(
        nn::reshape(nn::conv2d(p, "rpn/deltas", h, 4 * level.anchors_per_cell, 1, linear),
                    {batch, count, 4}));
  }
  r.objectness_logits = logits.size() == 1 ? logits[0] : nn::concat(logits, 1);
  r.deltas = deltas.size() == 1 ? deltas[0] : nn::concat(deltas, 1);
  r.objectness = nn::sigmoid(r.objectness_logits);
  return r;
}
/// @brief Combine sampled-anchor logistic loss with positive-anchor Smooth-L1 regression.
///
/// Uses softplus(logit)-logit*label, giving sigmoid(logit)-label at exactly
/// zero. Empty contributions have zero numerator and denominator one. No fitting
/// or mutation. Shape/dtype are checked; finite binary labels and nonnegative
/// weights are caller requirements. Nonfinite numerical inputs propagate.
///
/// @param rpn RPN result with [N,A] logits and [N,A,4] deltas.
/// @param labels Float32/64 [N,A] binary targets in the logits dtype.
/// @param weights Same-shaped nonnegative sampling weights; only weighted entries contribute.
/// @param targets Floating [N,A,4] normalized regression targets, matching deltas.
/// @param positives Same-shaped nonnegative positive-anchor weights; zeros disable regression.
/// @return Scalar sum of weighted BCE / max(sum(weights),1) and summed-coordinate regression / max(sum(positives),1).
/// @throws std::invalid_argument For mismatched shapes or nonfloating/mismatched dtypes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto r = n::models::BuildRpn(p, {{n::zeros({1, 1, 1, 4}), 4, 1, 2}}, 8);
/// const auto loss = n::models::RpnLoss(r, n::tensor({1., 0.}, {1, 2}), n::ones({1, 2}),
///                                      n::zeros({1, 2, 4}), n::zeros({1, 2}));
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see BuildRpn(), DetectorSmoothL1()
inline nn::Tensor RpnLoss(const Rpn &rpn, const nn::Tensor &labels, const nn::Tensor &weights,
                          const nn::Tensor &targets, const nn::Tensor &positives) {
  const auto &all = rpn.objectness_logits;
  if (all.dim() != 2 ||
      rpn.deltas.shape() != std::vector<std::int64_t>{all.shape()[0], all.shape()[1], 4} ||
      labels.shape() != all.shape() || weights.shape() != all.shape() ||
      positives.shape() != all.shape() || targets.shape() != rpn.deltas.shape() ||
      (all.dtype() != nn::DType::Float32 && all.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid RPN loss shapes");
  for (const auto &x : {rpn.deltas, labels, weights, targets, positives})
    if (x.dtype() != all.dtype())
      throw std::invalid_argument("RPN loss dtype mismatch");
  const auto one = nn::ones({}, all.dtype());
  const auto classification = nn::sum((nn::softplus(all) - all * labels) * weights) /
                              nn::maximum(nn::sum(weights), one);
  const auto regression =
      nn::sum(nn::sum(DetectorSmoothL1(rpn.deltas - targets), {2}) * positives) /
      nn::maximum(nn::sum(positives), one);
  return classification + regression;
}
/// @brief Build the baseline 28-pixel patch CNN with ten digit classes and background.
///
/// Biased SAME 3x3 ReLU convolutions of widths 32/64, SAME 2x2 pools,
/// a 128-unit dense ReLU and an 11-way classifier. No batch norm, dropout,
/// preprocessing or fitting. Input and store leaves must outlive backward.
///
/// @param p Mutable Float32/64 learned parameter store; leaves created before an optimizer is constructed.
/// @param image Floating [N,28,28,1] patch batch, N positive, same dtype as p.
/// @return Differentiable [N,11] linear logits; background is index 10.
/// @throws std::invalid_argument For input shape/dtype or incompatible existing store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto logits = n::models::PatchDetectorLogits(p, n::zeros({1, 28, 28, 1}));
/// @endcode
/// @par References
/// - Girshick et al. (2014), Rich feature hierarchies for accurate object detection and semantic segmentation, https://arxiv.org/abs/1311.2524
/// @see BuildRpn()
inline nn::Tensor PatchDetectorLogits(nn::ParameterStore &p, const nn::Tensor &image) {
  if (image.dim() != 4 || image.shape()[0] < 1 || image.shape()[1] != 28 ||
      image.shape()[2] != 28 || image.shape()[3] != 1 || image.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64))
    throw std::invalid_argument("patch detector requires floating [N,28,28,1]");
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h = nn::max_pool2d(nn::conv2d(p, "conv1", image, 32, 3, cfg), 2, 2, nn::Padding::Same);
  h = nn::max_pool2d(nn::conv2d(p, "conv2", h, 64, 3, cfg), 2, 2, nn::Padding::Same);
  return nn::dense(
      p, "output",
      nn::dense(p, "fc1", nn::reshape(h, {-1, 7 * 7 * 64}), 128, nn::Activation::Relu), 11);
}
} // namespace datamunge::nn::models
