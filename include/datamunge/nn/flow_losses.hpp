#pragma once
/// @file flow_losses.hpp
/// @brief Safe tensor masked multiscale smooth endpoint objectives in explicit units.
#include "datamunge/nn/flow_components.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute masked smooth endpoint error with per-image normalization.
///
/// Each valid error is sqrt(dx²+dy²+Float32(1e-6))-Float32(.001).
/// Empty images do not dilute the mean. Invalid targets/predictions are selected
/// out before arithmetic;their gradients are zero. Constants preserve source
/// Float32 rounding,so the zero floor may have a tiny signed residual.
///
/// @param prediction Floating positive[N,H,W,2] estimate.
/// @param target Matching dx/dy targets;nonfinite entries are excluded.
/// @param mask Numeric[N,H,W,1],included when>.5 and both target components finite.
/// @return Scalar mean over images with valid pixels;graph-connected zero when all masks are empty.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto p = n::zeros({2, 3, 4, 2}).requires_grad_();
/// const auto loss = m::FlowEndpointLoss(p, n::ones_like(p), n::ones({2, 3, 4, 1}));
/// const auto dx = n::grad(loss, {p});
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowEndpointLoss(const nn::Tensor &prediction, const nn::Tensor &target,
                                   const nn::Tensor &mask) {
  if (prediction.dim() != 4 || prediction.size(0) < 1 || prediction.size(1) < 1 ||
      prediction.size(2) < 1 || prediction.size(3) != 2 ||
      prediction.shape() != target.shape() ||
      mask.shape() != nn::Shape{prediction.size(0), prediction.size(1), prediction.size(2), 1})
    throw std::invalid_argument("aligned NHW2 endpoint supervision required");
  const nn::Tensor valid =
      nn::logical_and(nn::greater(mask, .5), nn::all(nn::is_finite(target), {3}, true));
  const nn::Tensor p = nn::where(valid, prediction, 0.0), t = nn::where(valid, target, 0.0);
  const nn::Tensor error =
      nn::sqrt(nn::sum(nn::squared_difference(p, t), {3}, true) + double(1e-6f)) -
      double(.001f);
  const nn::Tensor weights = nn::cast(valid, prediction.dtype());
  const nn::Tensor count = nn::sum(weights, {1, 2, 3});
  const nn::Tensor per_image =
      nn::sum(weights * error, {1, 2, 3}) / nn::maximum(count, nn::scalar_like(1.0, count));
  const nn::Tensor present = nn::cast(nn::greater(count, 0.0), prediction.dtype());
  const nn::Tensor images = nn::sum(present);
  return nn::sum(per_image * present) / nn::maximum(images, nn::scalar_like(1.0, images));
}

/// @brief Combine five masked endpoint objectives in fixed input-pixel/20 units.
///
/// Weights are Float32 .005,.01,.02,.08,.32,converted as source doubles.
/// Every pyramid head participates. Caller owns the explicit optimizer/minibatch
/// loop;this helper computes objectives only,no fitting or state mutation.
///
/// @param result Model pyramid fields ordered levels2..6.
/// @param target Matching five downsampled flow targets,input pixels/20.
/// @param mask Matching five one-channel validity masks.
/// @return Array{weighted total,level2,level3,level4,level5,level6} retaining all model gradients.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::FlowResult r;
/// std::array<n::Tensor, 5> truth, mask;
/// for (int i = 0; i < 5; ++i) {
///   r.pyramid[i] = n::zeros({1, 2, 2, 2});
///   truth[i] = n::ones({1, 2, 2, 2});
///   mask[i] = n::ones({1, 2, 2, 1});
/// }
/// const auto loss = m::FlowObjective(r, truth, mask);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline std::array<nn::Tensor, 6> FlowObjective(const FlowResult &result,
                                               const std::array<nn::Tensor, 5> &target,
                                               const std::array<nn::Tensor, 5> &mask) {
  /// @brief Source per-level loss coefficients for levels2..6.
  const float weights[] = {.005f, .01f, .02f, .08f, .32f};
  std::array<nn::Tensor, 6> losses;
  nn::Tensor total;
  for (int i = 0; i < 5; ++i) {
    losses[std::size_t(i) + 1] = FlowEndpointLoss(
        result.pyramid[std::size_t(i)], target[std::size_t(i)], mask[std::size_t(i)]);
    const nn::Tensor term = losses[std::size_t(i) + 1] * double(weights[i]);
    total = i ? total + term : term;
  }
  losses[0] = total;
  return losses;
}

} // namespace datamunge::nn::models
