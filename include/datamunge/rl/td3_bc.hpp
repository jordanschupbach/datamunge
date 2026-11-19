/// @file
/// @brief Behavior-regularized deterministic actor objective.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Combine deterministic policy improvement with an offline cloning penalty.
///
/// The value multiplier is alpha divided by the mean absolute policy
/// value, floored by epsilon. Stop its gradient so it scales the critic
/// signal without changing the direction of that signal.
/// @param actions Actor actions [batch,action_width], floating and nonempty.
/// @param demonstrated Same-shaped actions from the offline dataset, detached.
/// @param policy_values Critic scores [batch], matching the action dtype and device.
/// @param alpha Finite nonnegative critic weight, default 2.5.
/// @param epsilon Finite positive scale floor, default 1e-6.
/// @return Actor objective, detached value multiplier, and cloning mean squared error.
/// @throws std::invalid_argument Incompatible tensors or scaling parameters.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto out=dm::rl::td3_bc_actor_loss(n::zeros({2,1}),n::ones({2,1}),n::ones({2}));
/// std::print("{:.1f}\n",out[0].item()); // -1.5
/// @endcode
/// @par References
/// - Fujimoto and Gu (2021), A Minimalist Approach to Offline Reinforcement Learning.
///   https://arxiv.org/abs/2106.06860
inline std::array<nn::Tensor,3> td3_bc_actor_loss(const nn::Tensor& actions,
    const nn::Tensor& demonstrated,const nn::Tensor& policy_values,double alpha=2.5,
    double epsilon=1e-6) {
  namespace n=nn;
  if (!actions.defined() || actions.dim()!=2 || actions.numel()==0 || !n::is_floating(actions.dtype()) ||
      !demonstrated.defined() || !policy_values.defined() || demonstrated.shape()!=actions.shape() ||
      policy_values.shape()!=n::Shape{actions.size(0)} ||
      !(alpha>=0 && std::isfinite(alpha)) || !(epsilon>0 && std::isfinite(epsilon))) {
    throw std::invalid_argument("td3_bc_actor_loss: invalid shapes or scale");
  }
  for (const auto* field:{&demonstrated,&policy_values}) {
    if (field->dtype()!=actions.dtype() || field->device().backend!=actions.device().backend ||
        field->device().index!=actions.device().index) {
      throw std::invalid_argument("td3_bc_actor_loss: mismatched dtype or device");
    }
  }
  const auto floor=n::from_scalar(epsilon).to(actions.dtype()).to(actions.context());
  const auto scale=alpha/n::maximum(floor,n::mean(n::abs(policy_values.detach())));
  const auto cloning=n::mean(n::square(actions-demonstrated.detach()));
  return {cloning-scale*n::mean(policy_values),scale,cloning};
}

} // namespace datamunge::rl
