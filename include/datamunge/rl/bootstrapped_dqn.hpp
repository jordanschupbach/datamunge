/// @file
/// @brief Shared-trunk action values and masked temporal-difference loss.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Evaluate independent linear action heads above a shared ReLU layer.
/// @param parameters Store that owns the trunk and head parameters.
/// @param observations Float32 input [batch,features] matching the store dtype and context.
/// @param hidden Positive trunk width.
/// @param heads Positive number of independently parameterized output heads.
/// @param actions Positive number of action values per head.
/// @return Action values [batch,heads,actions].
/// @throws std::invalid_argument Incompatible input or layer sizes.
/// @par Example
/// @code
/// dm::nn::ParameterStore params(1);
/// const auto q=dm::rl::bootstrapped_values(params,dm::nn::eye(3),4,2,2);
/// std::print("{}\n",q.size(1)); // 2
/// @endcode
/// @par References
/// - Osband et al. (2016), Deep Exploration via Bootstrapped DQN.
///   https://arxiv.org/abs/1602.04621
/// @see bootstrapped_td_loss()
inline nn::Tensor bootstrapped_values(nn::ParameterStore& parameters,const nn::Tensor& observations,
                                       std::int64_t hidden,std::int64_t heads,std::int64_t actions) {
  if (!observations.defined() || observations.dim()!=2 || observations.size(1)<1 ||
      hidden<1 || heads<1 || actions<1) {
    throw std::invalid_argument("bootstrapped_values: invalid geometry");
  }
  const auto features=nn::dense(parameters,"shared",observations,hidden,nn::Activation::Relu);
  std::vector<nn::Tensor> values;
  values.reserve(heads);
  for (std::int64_t head=0;head<heads;++head) {
    values.push_back(nn::dense(parameters,"head"+std::to_string(head),features,actions));
  }
  return nn::stack(values,1);
}

/// @brief Compute masked squared TD error and detached per-head targets.
/// @param online Current action values Float32 CPU [batch,heads,actions].
/// @param target Successor action values Float32 CPU with the same shape.
/// @param actions Chosen action IDs [batch], CPU Int32 or Int64.
/// @param rewards Float32 CPU task rewards [batch].
/// @param terminal Float32 CPU terminal indicators [batch], zero or one.
/// @param mask Float32 CPU binary replay weights [batch,heads].
/// @param discount Finite discount in [0,1], default 0.95.
/// @return Scalar mean squared error over active entries and detached targets [batch,heads].
/// An all-zero mask yields exactly zero loss and output gradients for finite values.
/// @throws std::invalid_argument Invalid shape, data type, action index, binary mask or terminal code.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto q=n::zeros({1,2,2});
/// const auto [loss,td]=dm::rl::bootstrapped_td_loss(q,q,n::from_vector<int>({0}),
///     n::ones({1}),n::ones({1}),n::ones({1,2}));
/// std::print("{:.0f}\n",loss.item()); // 1
/// @endcode
/// @see bootstrapped_values()
inline std::pair<nn::Tensor,nn::Tensor> bootstrapped_td_loss(
    const nn::Tensor& online,const nn::Tensor& target,const nn::Tensor& actions,
    const nn::Tensor& rewards,const nn::Tensor& terminal,const nn::Tensor& mask,
    double discount=.95) {
  namespace n=nn;
  for (const auto* field:{&online,&target,&rewards,&terminal,&mask}) {
    if (!field->defined() || field->dtype()!=n::DType::Float32 ||
        field->device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("bootstrapped_td_loss: expected Float32 CPU values");
    }
  }
  if (online.dim()!=3 || online.numel()==0 || target.shape()!=online.shape() ||
      rewards.shape()!=n::Shape{online.size(0)} || terminal.shape()!=rewards.shape() ||
      mask.shape()!=n::Shape{online.size(0),online.size(1)} || !actions.defined() ||
      actions.shape()!=rewards.shape() || actions.device().backend!=compute::Backend::CPU ||
      (actions.dtype()!=n::DType::Int32 && actions.dtype()!=n::DType::Int64) ||
      !(discount>=0 && discount<=1)) {
    throw std::invalid_argument("bootstrapped_td_loss: incompatible shapes, actions or discount");
  }
  const auto ids=actions.to_vector<std::int64_t>();
  if (!std::all_of(ids.begin(),ids.end(),[&](auto id){return id>=0 && id<online.size(2);})) {
    throw std::invalid_argument("bootstrapped_td_loss: action out of range");
  }
  for (const auto* field:{&terminal,&mask}) {
    const auto values=field->to_vector<float>();
    if (!std::all_of(values.begin(),values.end(),[](float x){return x==0 || x==1;})) {
      throw std::invalid_argument("bootstrapped_td_loss: mask and terminal values must be binary");
    }
  }
  const auto selection=n::one_hot(actions,online.size(2)).reshape({online.size(0),1,online.size(2)});
  const auto chosen=n::sum(online*selection,{2});
  const auto td=(rewards.reshape({-1,1})+
      discount*(1.-terminal.reshape({-1,1}))*n::max(target.detach(),{2})).detach();
  return {n::sum(mask*n::square(chosen-td))/n::maximum(n::sum(mask),n::from_scalar(1.f)),td};
}

} // namespace datamunge::rl
