/// @file
/// @brief Additive and monotone mixing of decentralized action values.
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Sum decentralized utilities to obtain one value per joint action.
/// @param local Floating utilities [batch,agents], one selected action per agent.
/// @return Joint values [batch].
/// @throws std::invalid_argument Invalid utility shape or dtype.
/// @par Example
/// @code
/// std::print("{:.0f}\n",dm::rl::vdn_values(dm::nn::ones({1,2})).item()); // 2
/// @endcode
/// @par References
/// - Sunehag et al. (2018), Value-Decomposition Networks for Cooperative Multi-Agent Learning.
///   https://arxiv.org/abs/1706.05296
inline nn::Tensor vdn_values(const nn::Tensor& local) {
  if (!local.defined() || local.dim()!=2 || local.size(1)<1 || !nn::is_floating(local.dtype())) {
    throw std::invalid_argument("vdn_values: expected floating utility rows");
  }
  return nn::sum(local,{1});
}

/// @brief Use a state-conditioned monotone mixer to combine agent utilities.
///
/// Positive weights and an increasing ELU activation make the joint
/// value nondecreasing in each local utility.
/// @param params Mixer parameter store of matching dtype and context.
/// @param state Floating centralized state features [batch,features].
/// @param local Floating utilities [batch,agents] of matching dtype and device.
/// @param units Positive hidden mixer width, default eight.
/// @return Joint value [batch], first weights [batch,agents,units],
/// hidden bias [batch,units], second weights [batch,units], and output offset [batch,1].
/// @throws std::invalid_argument Invalid geometry, dtype, device or mixer width.
/// @par Example
/// @code
/// dm::nn::ParameterStore params(1);
/// const auto out=dm::rl::qmix_values(params,dm::nn::eye(2),dm::nn::ones({2,2}));
/// std::print("{}\n",out[0].size(0)); // 2
/// @endcode
/// @par References
/// - Rashid et al. (2018), QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.
///   https://proceedings.mlr.press/v80/rashid18a.html
/// @see vdn_values()
inline std::array<nn::Tensor,5> qmix_values(nn::ParameterStore& params,
    const nn::Tensor& state,const nn::Tensor& local,std::int64_t units=8) {
  namespace n=nn;
  if (!state.defined() || !local.defined() || state.dim()!=2 || local.dim()!=2 ||
      state.size(0)!=local.size(0) || state.size(1)<1 || local.size(1)<1 ||
      !n::is_floating(state.dtype()) || state.dtype()!=local.dtype() ||
      state.device().backend!=local.device().backend || state.device().index!=local.device().index ||
      units<1 || units>std::numeric_limits<std::int64_t>::max()/local.size(1)) {
    throw std::invalid_argument("qmix_values: incompatible inputs");
  }
  const auto batch=local.size(0),agents=local.size(1);
  const auto first=n::softplus(n::dense(params,"mix/w1",state,agents*units)).reshape({batch,agents,units});
  const auto second=n::softplus(n::dense(params,"mix/w2",state,units));
  const auto bias=n::dense(params,"mix/b1",state,units);
  const auto offset=n::dense(params,"mix/b2",state,1);
  const auto projected=n::matmul(local.reshape({batch,1,agents}),first).reshape({batch,units});
  const auto value=n::sum(n::elu(projected+bias)*second,{1})+offset.reshape({batch});
  return {value,first,bias,second,offset};
}

} // namespace datamunge::rl
