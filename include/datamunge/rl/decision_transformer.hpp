/// @file
/// @brief Return-conditioned causal action prediction for scalar trajectories.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Predict discrete actions from interleaved return, state and action tokens.
///
/// This scalar-observation variant embeds three tokens per step and adds
/// a shared learned time embedding. It reads action logits from state
/// token positions, so the policy has no access to the action at the
/// current step or future tokens. Reuse one store at a fixed horizon.
/// @param params Model parameter store with matching floating dtype.
/// @param returns Return-to-go values [batch,steps], floating CPU. Divided by steps before embedding.
/// @param states Scalar observations [batch,steps], numeric CPU.
/// @param actions Scalar action history [batch,steps], numeric CPU.
/// @param classes Positive number of discrete actions.
/// @param width Positive embedding width, default 24.
/// @param heads Positive attention head count dividing width, default 3.
/// @param blocks Positive number of decoder blocks, default 2.
/// @return Action logits [batch,steps,classes].
/// @throws std::invalid_argument Invalid shapes, storage or network options.
/// @par Example
/// @code
/// dm::nn::ParameterStore params(1);
/// const auto x=dm::nn::zeros({2,3});
/// const auto out=dm::rl::decision_transformer_logits(params,x,x,x,2);
/// std::print("{}\n",out.size(1)); // 3
/// @endcode
/// @par References
/// - Chen et al. (2021), Decision Transformer: Reinforcement Learning via Sequence Modeling.
///   https://arxiv.org/abs/2106.01345
inline nn::Tensor decision_transformer_logits(nn::ParameterStore& params,const nn::Tensor& returns,
    const nn::Tensor& states,const nn::Tensor& actions,std::int64_t classes,
    std::int64_t width=24,std::int64_t heads=3,int blocks=2) {
  namespace n=nn;
  if (!returns.defined() || returns.dim()!=2 || returns.numel()==0 || !n::is_floating(returns.dtype()) ||
      returns.device().backend!=compute::Backend::CPU || !states.defined() || !actions.defined() ||
      states.shape()!=returns.shape() || actions.shape()!=returns.shape() ||
      states.device().backend!=compute::Backend::CPU || actions.device().backend!=compute::Backend::CPU ||
      classes<1 || width<1 || heads<1 || width%heads!=0 || blocks<1 ||
      width>std::numeric_limits<std::int64_t>::max()/2 ||
      returns.size(1)>std::numeric_limits<std::int64_t>::max()/3) {
    throw std::invalid_argument("decision_transformer_logits: invalid inputs or network geometry");
  }
  const auto batch=returns.size(0),steps=returns.size(1),tokens=3*steps;
  const auto embed=[&](const n::Tensor& x,const char* name) {
    return n::dense(params,name,x.to(returns.dtype()).reshape({batch,steps,1}),width);
  };
  const auto reward=embed(returns/steps,"return"), state=embed(states,"state"), action=embed(actions,"action");
  const auto time=params.get("time",{1,steps,1,width},n::init::normal(.08));
  auto x=(n::stack({reward,state,action},2)+time).reshape({batch,tokens,width});
  const auto mask=n::causal_mask(tokens,returns.dtype());
  for (int i=0;i<blocks;++i) {
    const auto name="block"+std::to_string(i);
    const auto norm=n::layer_norm(params,name+"/ln1",x);
    const auto residual=x+n::multi_head_attention(params,name+"/attention",norm,norm,heads,{.mask=mask}).output;
    const auto hidden=n::dense(params,name+"/up",n::layer_norm(params,name+"/ln2",residual),
                               2*width,n::Activation::GeluTanh);
    x=residual+n::dense(params,name+"/down",hidden,width);
  }
  const auto normalized=n::layer_norm(params,"final",x);
  std::vector<std::int64_t> positions;
  for (std::int64_t t=0;t<steps;++t) positions.push_back(3*t+1);
  return n::dense(params,"policy",n::gather(normalized,n::from_vector(positions),1),classes);
}

} // namespace datamunge::rl
