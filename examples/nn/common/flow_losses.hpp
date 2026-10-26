// Safe masked multi-scale smooth endpoint supervision in input-image
// pixels/20: a port of tflow's common/ml/flow_losses.h (flownets, pwcnet).
#pragma once

#include <array>

#include "common/flow_ops.hpp"

namespace flow {

/// Per-image mean of sqrt(|p - t|^2 + 1e-6) - .001 over valid pixels (mask > .5
/// and finite target), averaged over images that have any valid pixel.
inline nn::Tensor FlowEndpointLoss(const nn::Tensor& prediction, const nn::Tensor& target, const nn::Tensor& mask) {
  const nn::Tensor valid = nn::logical_and(nn::greater(mask, .5), nn::all(nn::is_finite(target), {3}, true));
  const nn::Tensor p = nn::where(valid, prediction, 0.0), t = nn::where(valid, target, 0.0);
  const nn::Tensor error   = nn::sqrt(nn::sum(nn::squared_difference(p, t), {3}, true) + double(1e-6f)) - double(.001f);
  const nn::Tensor weights = nn::cast(valid, prediction.dtype());
  const nn::Tensor count   = nn::sum(weights, {1, 2, 3});
  const nn::Tensor per_image = nn::sum(weights * error, {1, 2, 3}) / nn::maximum(count, nn::scalar_like(1.0, count));
  const nn::Tensor present   = nn::cast(nn::greater(count, 0.0), prediction.dtype());
  const nn::Tensor images    = nn::sum(present);
  return nn::sum(per_image * present) / nn::maximum(images, nn::scalar_like(1.0, images));
}

/// {weighted total, level2..level6} with weights .005/.01/.02/.08/.32.
inline std::array<nn::Tensor, 6> FlowObjective(const FlowResult& result, const std::array<nn::Tensor, 5>& target,
                                                const std::array<nn::Tensor, 5>& mask) {
  const float               weights[] = {.005f, .01f, .02f, .08f, .32f};
  std::array<nn::Tensor, 6> losses;
  nn::Tensor                total;
  for (int i = 0; i < 5; ++i) {
    losses[std::size_t(i) + 1] = FlowEndpointLoss(result.pyramid[std::size_t(i)], target[std::size_t(i)], mask[std::size_t(i)]);
    const nn::Tensor term      = losses[std::size_t(i) + 1] * double(weights[i]);
    total                      = i ? total + term : term;
  }
  losses[0] = total;
  return losses;
}

}  // namespace flow
