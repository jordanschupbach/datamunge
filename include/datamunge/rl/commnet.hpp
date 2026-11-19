/// @file
/// @brief Shared multi-agent communication policy.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Apply two rounds of mean peer communication to private observations.
///
/// Each agent receives the mean hidden state of all other agents. All
/// agents share network weights. The gain scales messages before they
/// enter each update, allowing a muted-channel evaluation with zero gain.
/// @param params Float32 parameter store for the shared policy.
/// @param observations Float32 CPU input [batch,agents,features], with at least two agents.
/// @param gain Float32 CPU channel gains [hidden].
/// @param actions Positive number of action logits per agent.
/// @return Action logits, first-round hidden state and peer messages,
/// then second-round hidden state and peer messages.
/// @throws std::invalid_argument Invalid input or network geometry.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// n::ParameterStore params(1);
/// const auto output=dm::rl::commnet_policy(params,n::zeros({1,3,2}),n::ones({4}),2);
/// std::print("{}\n",output[0].size(1)); // 3
/// @endcode
/// @par References
/// - Sukhbaatar, Szlam and Fergus (2016), Learning Multiagent Communication with Backpropagation.
///   https://arxiv.org/abs/1605.07736
inline std::array<nn::Tensor,5> commnet_policy(nn::ParameterStore& params,
    const nn::Tensor& observations,const nn::Tensor& gain,std::int64_t actions) {
  namespace n=nn;
  if (!observations.defined() || observations.dim()!=3 || observations.numel()==0 ||
      observations.size(1)<2 || observations.dtype()!=n::DType::Float32 ||
      observations.device().backend!=compute::Backend::CPU || !gain.defined() ||
      gain.dim()!=1 || gain.size(0)<1 || gain.size(0)>std::numeric_limits<std::int64_t>::max()/3 ||
      gain.dtype()!=n::DType::Float32 || gain.device().backend!=compute::Backend::CPU || actions<1) {
    throw std::invalid_argument("commnet_policy: incompatible inputs");
  }
  const auto encoded=n::dense(params,"encode",observations,gain.size(0),n::Activation::Tanh);
  auto hidden=encoded;
  std::array<n::Tensor,5> result;
  for (int round=0;round<2;++round) {
    const auto peers=(n::sum(hidden,{1},true)-hidden)/(observations.size(1)-1);
    result[1+2*round]=hidden;
    result[2+2*round]=peers;
    hidden=n::dense(params,"round"+std::to_string(round),n::concat({hidden,peers*gain,encoded},2),
                    gain.size(0),n::Activation::Tanh);
  }
  result[0]=n::dense(params,"action",hidden,actions);
  return result;
}

} // namespace datamunge::rl
