/// @file
/// @brief Recurrent trial policy with explicit episode-boundary memory control.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Unroll a recurrent categorical policy through a meta-RL trial.
///
/// The GRU starts at zero. By default it retains hidden state at
/// episode boundaries. Setting carry to false resets the hidden state
/// before processing each new episode's first input.
/// @param params Model parameters shared across all time steps.
/// @param histories Float32 CPU inputs [batch,steps,features], with positive dimensions.
/// @param hidden Positive hidden-state width.
/// @param actions Positive action count.
/// @param episode_steps Positive number of steps between episode boundaries.
/// @param carry Whether to preserve hidden state across episodes, default true.
/// @return Action log probabilities [batch,steps,actions] and hidden states [batch,steps,hidden].
/// @throws std::invalid_argument Invalid input or geometry.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(1);
/// const auto [logp,hidden]=dm::rl::rl_squared_policy(p,dm::nn::zeros({2,4,4}),24,2,2);
/// std::print("{}\n",logp.size(1)); // 4
/// @endcode
/// @par References
/// - Duan et al. (2016), RL²: Fast Reinforcement Learning via Slow Reinforcement Learning.
///   https://arxiv.org/abs/1611.02779
inline std::pair<nn::Tensor,nn::Tensor> rl_squared_policy(nn::ParameterStore& params,
    const nn::Tensor& histories,std::int64_t hidden,std::int64_t actions,
    std::int64_t episode_steps,bool carry=true) {
  namespace n=nn;
  if (!histories.defined() || histories.dim()!=3 || histories.numel()==0 ||
      histories.dtype()!=n::DType::Float32 || histories.device().backend!=compute::Backend::CPU ||
      hidden<1 || actions<1 || episode_steps<1) {
    throw std::invalid_argument("rl_squared_policy: invalid trial geometry");
  }
  const auto batch=histories.size(0),width=histories.size(2);
  auto state=n::zeros({batch,hidden});
  std::vector<n::Tensor> policy,states;
  for (std::int64_t step=0;step<histories.size(1);++step) {
    if (!carry && step>0 && step%episode_steps==0) state=n::zeros_like(state);
    const auto input=n::slice(histories,{0,step,0},{batch,1,width}).reshape({batch,width});
    state=n::gru_step(params,"cell",input,state);
    states.push_back(state);
    policy.push_back(n::log_softmax(n::dense(params,"policy",state,actions),-1));
  }
  return {n::stack(policy,1),n::stack(states,1)};
}

} // namespace datamunge::rl
