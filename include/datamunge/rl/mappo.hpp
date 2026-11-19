/// @file
/// @brief Clipped multi-agent actor objective with centralized value regression.
#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Compute a MAPPO objective with separate ratios and a shared team advantage.
///
/// Each agent's selected-action ratio is clipped independently. The
/// critic contributes a single value error per team observation, and
/// the entropy term averages over all agents. Rollout quantities are detached.
/// @param logp Current action log probabilities [batch,agents,actions], floating CPU.
/// @param actions CPU Int32 or Int64 executed actions [batch,agents].
/// @param old Selected log probabilities from collection [batch,agents].
/// @param advantage Team advantage [batch].
/// @param value Current centralized value estimate [batch].
/// @param target Team critic regression target [batch].
/// @param clip Ratio clipping radius in (0,1), default 0.2.
/// @param value_weight Finite nonnegative critic weight, default 0.5.
/// @param entropy_weight Finite nonnegative entropy weight, default 0.01.
/// @return Total loss, clipped policy loss, and per-agent ratios.
/// @throws std::invalid_argument Invalid shapes, options, storage or action labels.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto zero=n::zeros({1});
/// const auto out=dm::rl::mappo_loss(n::zeros({1,2,1}),n::zeros({1,2},n::DType::Int32),
///     n::zeros({1,2}),n::ones({1}),zero,zero);
/// std::print("{:.0f}\n",out[1].item()); // -1
/// @endcode
/// @par References
/// - Yu et al. (2022), The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
///   https://arxiv.org/abs/2103.01955
inline std::array<nn::Tensor,3> mappo_loss(const nn::Tensor& logp,const nn::Tensor& actions,
    const nn::Tensor& old,const nn::Tensor& advantage,const nn::Tensor& value,const nn::Tensor& target,
    double clip=.2,double value_weight=.5,double entropy_weight=.01) {
  namespace n=nn;
  for (const auto* field:{&logp,&old,&advantage,&value,&target}) {
    if (!field->defined() || !n::is_floating(field->dtype()) || field->dtype()!=logp.dtype() ||
        field->device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("mappo_loss: expected matching floating CPU tensors");
    }
  }
  if (logp.dim()!=3 || logp.numel()==0 || old.shape()!=n::Shape{logp.size(0),logp.size(1)} ||
      !actions.defined() || actions.shape()!=old.shape() || actions.device().backend!=compute::Backend::CPU ||
      (actions.dtype()!=n::DType::Int32 && actions.dtype()!=n::DType::Int64) ||
      value.shape()!=n::Shape{logp.size(0)} || advantage.shape()!=value.shape() || target.shape()!=value.shape() ||
      !(clip>0 && clip<1) || !(value_weight>=0 && std::isfinite(value_weight)) ||
      !(entropy_weight>=0 && std::isfinite(entropy_weight))) {
    throw std::invalid_argument("mappo_loss: incompatible tensors or options");
  }
  for (const auto action:actions.to_vector<std::int64_t>()) {
    if (action<0 || action>=logp.size(2)) throw std::invalid_argument("mappo_loss: action out of range");
  }
  const auto chosen=n::sum(logp*n::one_hot(actions,logp.size(2),1.,0.,logp.dtype()),{2});
  const auto ratio=n::exp(chosen-old.detach());
  const auto gain=advantage.detach().reshape({-1,1});
  const auto policy=-n::mean(n::minimum(gain*ratio,gain*n::clip(ratio,1.-clip,1.+clip)));
  const auto error=n::mean(n::square(value-target.detach()));
  const auto neg_entropy=n::mean(n::sum(n::exp(logp)*logp,{2}));
  return {policy+value_weight*error+entropy_weight*neg_entropy,policy,ratio};
}

} // namespace datamunge::rl
