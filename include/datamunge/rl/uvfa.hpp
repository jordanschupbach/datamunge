/// @file
/// @brief Coordinate-conditioned action-value approximation.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Predict bounded action values from states and relative goal coordinates.
///
/// Concatenates the scaled state coordinates and goal displacement, then
/// applies two shared ReLU layers and a sigmoid action head. One parameter
/// store is reused across all goals. The output suits targets in [0,1].
/// @param params Parameter store reused for training and evaluation.
/// @param states Floating tensor [batch,coordinates] compatible with the store.
/// @param goals Goal coordinates with the same geometry, dtype and device.
/// @param coordinate_scale Finite positive divisor for both feature blocks.
/// @param hidden Positive width of both hidden layers.
/// @param actions Positive number of action values.
/// @return Action value rows [batch,actions].
/// @throws std::invalid_argument Invalid tensor geometry or model sizes.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// n::ParameterStore params(1);
/// const auto values=dm::rl::universal_action_values(params,n::zeros({2,2}),
///     n::ones({2,2}),6.,48,4);
/// std::print("{}\n",values.size(1)); // 4
/// @endcode
/// @par References
/// - Schaul et al. (2015), Universal Value Function Approximators.
///   https://proceedings.mlr.press/v37/schaul15.html
inline nn::Tensor universal_action_values(nn::ParameterStore& params,const nn::Tensor& states,
    const nn::Tensor& goals,double coordinate_scale,std::int64_t hidden,std::int64_t actions) {
  namespace n=nn;
  if (!states.defined() || !goals.defined() || states.dim()!=2 || states.size(1)<1 ||
      goals.shape()!=states.shape() || !n::is_floating(states.dtype()) ||
      goals.dtype()!=states.dtype() || goals.device().backend!=states.device().backend ||
      goals.device().index!=states.device().index ||
      !(coordinate_scale>0 && std::isfinite(coordinate_scale)) || hidden<1 || actions<1) {
    throw std::invalid_argument("universal_action_values: incompatible coordinates or model sizes");
  }
  const auto features=n::concat({states/coordinate_scale,(goals-states)/coordinate_scale},1);
  const auto first=n::dense(params,"hidden1",features,hidden,n::Activation::Relu);
  const auto second=n::dense(params,"hidden2",first,hidden,n::Activation::Relu);
  return n::dense(params,"values",second,actions,n::Activation::Sigmoid);
}

} // namespace datamunge::rl
