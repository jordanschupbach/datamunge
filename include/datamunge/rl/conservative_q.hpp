/// @file
/// @brief Discrete conservative Q-learning with empirical action counts.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Add an empirical conservative penalty to a squared Bellman objective.
///
/// The Bellman term averages half squared TD error under empirical
/// state-action weights. The penalty averages log-sum-exp over actions
/// under the empirical state distribution and subtracts the empirical
/// average Q-value. Targets and counts are detached.
/// @param values Finite Float32 or Float64 CPU Q-values [states,actions].
/// @param targets Same-shaped TD targets on the same device and dtype.
/// @param counts Same-shaped nonnegative finite empirical counts with positive total.
/// @param alpha Finite nonnegative conservative penalty weight, default 0.5.
/// @return Total objective, squared Bellman term and unweighted conservative penalty.
/// @throws std::invalid_argument Invalid geometry, counts or penalty weight.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto q=n::zeros({1,2});
/// const auto out=dm::rl::conservative_q_loss(q,n::ones({1,2}),n::ones({1,2}),0.);
/// std::print("{:.1f}\n",out[0].item()); // 0.5
/// @endcode
/// @par References
/// - Kumar et al. (2020), Conservative Q-Learning for Offline Reinforcement Learning.
///   https://arxiv.org/abs/2006.04779
inline std::array<nn::Tensor,3> conservative_q_loss(const nn::Tensor& values,
    const nn::Tensor& targets,const nn::Tensor& counts,double alpha=.5) {
  namespace n=nn;
  for (const auto* field:{&values,&targets,&counts}) {
    if (!field->defined() || field->dtype()!=values.dtype() ||
        field->device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("conservative_q_loss: expected matching CPU tensors");
    }
  }
  if ((values.dtype()!=n::DType::Float32 && values.dtype()!=n::DType::Float64) ||
      values.dim()!=2 || values.numel()==0 || targets.shape()!=values.shape() ||
      counts.shape()!=values.shape() || !(alpha>=0 && std::isfinite(alpha))) {
    throw std::invalid_argument("conservative_q_loss: invalid geometry or alpha");
  }
  for (const auto count:counts.to_vector<double>()) {
    if (!(count>=0 && std::isfinite(count))) {
      throw std::invalid_argument("conservative_q_loss: invalid action count");
    }
  }
  const auto detached=counts.detach();
  const auto total=n::sum(detached);
  if (!(total.item()>0 && std::isfinite(total.item()))) {
    throw std::invalid_argument("conservative_q_loss: invalid count total");
  }
  const auto weight=detached/total;
  const auto bellman=.5*n::sum(weight*n::square(values-targets.detach()));
  const auto penalty=n::sum(n::sum(weight,{1})*n::logsumexp(values,{1}))-n::sum(weight*values);
  return {bellman+alpha*penalty,bellman,penalty};
}

} // namespace datamunge::rl
