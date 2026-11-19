/// @file
/// @brief Differentiable one-step policy adaptation for contextual bandits.
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Exact support-event expectations for a batch of contextual bandit tasks.
struct MamlBanditEvaluation {
  nn::Tensor pre_update;  ///< Scalar expected query reward before adaptation.
  nn::Tensor post_update; ///< Scalar expected query reward after adaptation.
  nn::Tensor adapted;     ///< Adapted weights [tasks * contexts * 4, features].
  nn::Tensor mass;        ///< Probability of each event within its task [events].
  nn::Tensor base;        ///< Pre-update query reward by support event [events].
  nn::Tensor after;       ///< Post-update query reward by support event [events].
};

/// @brief Evaluate one exact MAML policy-gradient step for binary contextual bandits.
///
/// Contexts and tasks are uniform. Each task gives a Bernoulli reward probability
/// for each context and action. Events are ordered by task, context, action, reward.
/// Differentiating post_update includes both the support policy likelihood and
/// the derivative of the inner policy-gradient step.
/// @param weights Initial logistic policy weights [features].
/// @param features Context feature matrix [contexts, features].
/// @param tasks Reward probabilities in [0,1], shaped [tasks, contexts, 2].
/// @param inner_rate Positive step size applied to reward centered at 0.5.
/// @return Differentiable query expectations and event-level diagnostics.
/// @throws std::invalid_argument Invalid tensor geometry or step size.
/// @see MamlBanditEvaluation
inline MamlBanditEvaluation maml_bandit(const nn::Tensor& weights, const nn::Tensor& features,
                                       const nn::Tensor& tasks, double inner_rate = 2.0) {
  namespace n = nn;
  if (!weights.defined() || !features.defined() || !tasks.defined() ||
      weights.dim() != 1 || features.dim() != 2 || tasks.dim() != 3 ||
      weights.numel() == 0 || tasks.numel() == 0 || features.size(1) != weights.size(0) ||
      features.size(0) != tasks.size(1) || tasks.size(2) != 2 ||
      !std::isfinite(inner_rate) || inner_rate <= 0) {
    throw std::invalid_argument("maml_bandit: invalid bandit geometry or step size");
  }
  const auto count = tasks.size(0), contexts = features.size(0), width = weights.size(0);
  const auto low = n::slice(tasks, {0,0,0}, {count,contexts,1}).reshape({count,contexts});
  const auto high = n::slice(tasks, {0,0,1}, {count,contexts,1}).reshape({count,contexts});
  const auto prior = n::sigmoid(n::matmul(weights.reshape({1,width}), features, false, true));
  const auto unadapted = n::mean(low + prior * (high - low), {1});
  std::vector<n::Tensor> adapted, masses, before, after;
  for (std::int64_t context = 0; context < contexts; ++context) {
    const auto x = n::slice(features, {context,0}, {1,width});
    const auto p = n::slice(prior, {0,context}, {1,1});
    for (int action = 0; action < 2; ++action) {
      const auto table = action ? high : low;
      const auto env = n::slice(table, {0,context}, {count,1}).reshape({count});
      const auto likelihood = (action ? p : 1.0 - p).reshape({});
      for (int reward = 0; reward < 2; ++reward) {
        const auto updated = weights + (inner_rate * (reward - 0.5)) * (action - p) * x;
        const auto posterior = n::sigmoid(n::matmul(updated, features, false, true));
        const auto query = n::mean(low + posterior * (high - low), {1});
        masses.push_back(likelihood * (reward ? env : 1.0 - env) / contexts);
        adapted.push_back(n::broadcast_to(updated, {count,width}));
        before.push_back(unadapted);
        after.push_back(query);
      }
    }
  }
  const auto mass = n::stack(masses, 1);
  const auto base = n::stack(before, 1);
  const auto query = n::stack(after, 1);
  return {n::mean(n::sum(mass * base, {1})), n::mean(n::sum(mass * query, {1})),
          n::stack(adapted, 1).reshape({-1,width}), mass.reshape({-1}),
          base.reshape({-1}), query.reshape({-1})};
}

} // namespace datamunge::rl
