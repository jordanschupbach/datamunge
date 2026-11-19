/// @file
/// @brief Box-constrained input stress tests and categorical prediction divergence.
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Input states and derivatives from projected sign ascent.
struct ProjectedSignTrace {
  Tensor states;    ///< Initial and updated states, stacked on a new leading axis.
  Tensor gradients; ///< Ascent gradients before projection, stacked on a new leading axis.
};

/// @brief Ascend a scalar input objective within a unit-box L-infinity neighborhood.
/// @param reference Clean inputs in [0,1].
/// @param start Starting inputs with the same shape, dtype and device.
/// @param objective Differentiable scalar score to maximize.
/// @param radius Nonnegative L-infinity displacement bound relative to reference.
/// @param rate Positive update size.
/// @param steps Positive number of updates.
/// @return Detached states and gradients, including the projected initial state.
/// @throws std::invalid_argument Invalid settings or mismatched input geometry.
inline ProjectedSignTrace projected_sign_ascent(const Tensor& reference, const Tensor& start,
    const std::function<Tensor(const Tensor&)>& objective, double radius, double rate, int steps) {
  if (!reference.defined() || !start.defined() || reference.numel() == 0 ||
      reference.shape() != start.shape() || reference.dtype() != start.dtype() ||
      reference.device().backend != start.device().backend ||
      reference.device().index != start.device().index || !objective || steps < 1 ||
      !std::isfinite(radius) || radius < 0 ||
      !std::isfinite(rate) || rate <= 0) {
    throw std::invalid_argument("projected_sign_ascent: invalid inputs or settings");
  }
  const auto low = clip(reference.detach() - radius, 0.0, 1.0);
  const auto high = clip(reference.detach() + radius, 0.0, 1.0);
  auto current = clip(start.detach(), low, high);
  std::vector<Tensor> states{current}, gradients;
  for (int step = 0; step < steps; ++step) {
    auto leaf = current.detach();
    leaf.requires_grad_(true);
    const auto gradient = grad(objective(leaf), {leaf}).at(0).detach();
    current = clip(current + rate * sign(gradient), low, high);
    gradients.push_back(gradient);
    states.push_back(current);
  }
  return {stack(states), stack(gradients)};
}

/// @brief Mean categorical KL divergence from reference logits to comparison logits.
/// @param reference Nonempty [batch,classes] reference logits.
/// @param comparison Comparison logits with the same shape.
/// @return Scalar KL divergence with gradients through both sets of logits.
/// @throws std::invalid_argument Invalid logit geometry.
inline Tensor categorical_logit_kl(const Tensor& reference, const Tensor& comparison) {
  if (!reference.defined() || !comparison.defined() || reference.dim() != 2 ||
      reference.numel() == 0 || reference.shape() != comparison.shape()) {
    throw std::invalid_argument("categorical_logit_kl: expected matching logit matrices");
  }
  const auto logp = log_softmax(reference, 1);
  return mean(sum(exp(logp) * (logp - log_softmax(comparison, 1)), {1}));
}

} // namespace datamunge::nn
