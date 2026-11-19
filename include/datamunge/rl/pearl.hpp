/// @file
/// @brief Probabilistic context inference and soft actor-critic for a binary bandit.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Posterior and training diagnostics for a PEARL bandit batch.
struct PearlBanditEvaluation {
  nn::Tensor mean;       ///< Posterior mean [batch,1].
  nn::Tensor variance;   ///< Posterior variance [batch,1].
  nn::Tensor latent;     ///< Reparameterized task sample [batch,1].
  nn::Tensor policy;     ///< Action probabilities [batch,2].
  nn::Tensor q1;         ///< First critic predictions [batch,2].
  nn::Tensor q2;         ///< Second critic predictions [batch,2].
  nn::Tensor actor_loss; ///< Mean entropy-regularized actor objective.
  nn::Tensor loss;       ///< Combined actor, critics and posterior KL objective.
};

/// @brief Evaluate PEARL's discrete bandit objective on off-policy replay.
///
/// Each selected transition contributes a scalar Gaussian factor. An empty
/// selection falls back to the standard normal prior. The two critics train
/// against immediate rewards and backpropagate through the sampled latent.
/// The actor treats the latent and the minimum critic value as fixed.
/// @param parameters Model weights, initialized on the first call.
/// @param context Action and reward pairs [batch,steps,2].
/// @param mask Binary context selection [batch,steps], stored as floating point.
/// @param noise Standard normal draws [batch,1].
/// @param action Replay action indices [batch] taking values zero or one.
/// @param reward Observed replay rewards [batch].
/// @param width Positive hidden-layer width.
/// @param temperature Nonnegative entropy coefficient.
/// @param kl_weight Nonnegative posterior KL penalty coefficient.
/// @return Differentiable loss and inference diagnostics.
/// @throws std::invalid_argument Incompatible tensor shapes or invalid coefficients.
inline PearlBanditEvaluation pearl_bandit(nn::ParameterStore& parameters,
    const nn::Tensor& context, const nn::Tensor& mask, const nn::Tensor& noise,
    const nn::Tensor& action, const nn::Tensor& reward, std::int64_t width = 16,
    double temperature = .1, double kl_weight = .01) {
  namespace n = nn;
  if (!context.defined() || context.dim() != 3 || context.numel() == 0 || context.size(2) != 2 ||
      !mask.defined() || !noise.defined() || !action.defined() || !reward.defined() ||
      width < 1 || !std::isfinite(temperature) || temperature < 0 ||
      !std::isfinite(kl_weight) || kl_weight < 0) {
    throw std::invalid_argument("pearl_bandit: invalid replay tensors or coefficients");
  }
  const auto batch = context.size(0), steps = context.size(1), rows = batch * steps;
  if (mask.shape() != n::Shape{batch,steps} || noise.shape() != n::Shape{batch,1} ||
      action.shape() != n::Shape{batch} || reward.shape() != n::Shape{batch}) {
    throw std::invalid_argument("pearl_bandit: incompatible replay shapes");
  }
  const auto flat = context.reshape({rows,2});
  const auto a = n::slice(flat, {0,0}, {rows,1}).reshape({rows});
  const auto r = n::slice(flat, {0,1}, {rows,1}).reshape({rows});
  const auto features = n::stack({a,r,(2.0*a-1.0)*(2.0*r-1.0)}, 1);
  const auto network = [&](const n::Tensor& input, const std::string& hidden, const std::string& output) {
    return n::dense(parameters, output, n::tanh(n::dense(parameters, hidden, input, width)), 2);
  };
  const auto factors = network(features, "encoder_hidden", "factors");
  const auto means = n::slice(factors, {0,0}, {rows,1}).reshape({batch,steps});
  const auto raw = n::slice(factors, {0,1}, {rows,1}).reshape({batch,steps});
  const auto precision = mask * (.01 + n::softplus(raw));
  const auto total = n::sum(precision, {1}, true);
  const auto variance = 1.0 / n::where(n::greater(total, 0.0), total, 1.0);
  const auto mean = variance * n::sum(precision * means, {1}, true);
  const auto latent = mean + n::sqrt(variance) * noise;
  const auto q1 = network(latent, "q1_hidden", "q1");
  const auto q2 = network(latent, "q2_hidden", "q2");
  const auto selected = n::one_hot(action, 2);
  const auto critic = [&](const n::Tensor& q) {
    return n::mean(n::square(n::sum(q * selected, {1}) - reward));
  };
  const auto logp = n::log_softmax(network(latent.detach(), "actor_hidden", "actor"), -1);
  const auto policy = n::exp(logp);
  const auto actor_loss = n::mean(n::sum(policy * (temperature * logp - n::minimum(q1,q2).detach()), {1}));
  const auto kl = .5 * n::mean(n::square(mean) + variance - 1.0 - n::log(variance));
  return {mean, variance, latent, policy, q1, q2, actor_loss,
          critic(q1) + critic(q2) + actor_loss + kl_weight * kl};
}

} // namespace datamunge::rl
