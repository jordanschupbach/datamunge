/// @file
/// @brief Single-transition Option-Critic objectives.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Compute critic, intra-option actor and termination loss for one transition.
///
/// The critic target mixes continuing the current option with switching
/// to the best next option. The actor uses a detached TD error, and the
/// termination objective uses detached continuation advantage plus a
/// margin. Terminal transitions bootstrap zero and train no termination.
/// @param current Scalar value of the chosen current option.
/// @param successor Vector of next-state option values.
/// @param policy_logits Vector of intra-option action logits.
/// @param termination_logit Scalar next-state termination logit for the current option.
/// @param option Current option index in successor.
/// @param action Chosen action index in policy_logits.
/// @param reward Finite immediate reward.
/// @param terminal Whether the transition ends the episode.
/// @param discount Finite discount in [0,1], default 0.95.
/// @param margin Finite nonnegative termination margin, default 0.02.
/// @return Scalar loss, termination probability, TD error and continuation advantage.
/// Tensors must have matching floating dtype and CPU storage.
/// @throws std::invalid_argument Incompatible tensors, indices or scalar options.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto out=dm::rl::option_critic_loss(n::zeros({}),n::zeros({2}),
///     n::zeros({2}),n::zeros({}),0,1,1.,true);
/// std::print("{:.0f}\n",out[2].item()); // 1
/// @endcode
/// @par References
/// - Bacon, Harb and Precup (2017), The Option-Critic Architecture.
///   https://arxiv.org/abs/1609.05140
inline std::array<nn::Tensor,4> option_critic_loss(
    const nn::Tensor& current,const nn::Tensor& successor,const nn::Tensor& policy_logits,
    const nn::Tensor& termination_logit,std::int64_t option,std::int64_t action,
    double reward,bool terminal,double discount=.95,double margin=.02) {
  namespace n=nn;
  for (const auto* value:{&current,&successor,&policy_logits,&termination_logit}) {
    if (!value->defined() || value->dtype()!=current.dtype() ||
        value->device().backend!=compute::Backend::CPU || !n::is_floating(value->dtype())) {
      throw std::invalid_argument("option_critic_loss: expected matching floating CPU tensors");
    }
  }
  if (current.dim()!=0 || termination_logit.dim()!=0 || successor.dim()!=1 ||
      policy_logits.dim()!=1 || option<0 || option>=successor.size(0) ||
      action<0 || action>=policy_logits.size(0) || !std::isfinite(reward) ||
      !(discount>=0 && discount<=1) || !(margin>=0 && std::isfinite(margin))) {
    throw std::invalid_argument("option_critic_loss: invalid shapes, indices or options");
  }
  const auto continuation=n::gather(successor,n::from_scalar(option));
  const auto best=n::max(successor);
  const auto advantage=(continuation-best).detach();
  const auto beta=n::sigmoid(termination_logit);
  const auto target=terminal?n::from_scalar(reward).to(current.dtype()):
      (reward+discount*((1.-beta)*continuation+beta*best)).detach();
  const auto td=target-current;
  const auto log_probability=n::gather(n::log_softmax(policy_logits,0),n::from_scalar(action));
  const auto loss=.5*n::square(td)-log_probability*td.detach()+
      (terminal?0.:1.)*beta*(advantage+margin);
  return {loss,beta,td,advantage};
}

} // namespace datamunge::rl
