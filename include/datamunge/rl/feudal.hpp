/// @file
/// @brief Directional manager and goal-conditioned worker operations.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

namespace detail {
/// @brief Normalize row vectors, flooring the squared norm at 1e-8.
/// @param rows Floating coordinate matrix.
/// @return Row directions, with zero rows preserved.
inline nn::Tensor feudal_unit(const nn::Tensor& rows) {
  return rows/nn::sqrt(nn::maximum(nn::sum(nn::square(rows),{1},true),
      nn::from_scalar(1e-8).to(rows.dtype())));
}
} // namespace detail

/// @brief Predict one direction per state using a tanh manager network.
/// @param params Manager parameter store or shared store with worker names.
/// @param states Floating state matrix [batch,coordinates].
/// @param hidden Positive hidden layer width.
/// @return Directions [batch,coordinates], normalized except for near-zero rows.
/// @throws std::invalid_argument Invalid state shape or layer width.
/// @par Example
/// @code
/// dm::nn::ParameterStore params(1);
/// const auto goal=dm::rl::feudal_direction(params,dm::nn::eye(2),16);
/// std::print("{}\n",goal.size(1)); // 2
/// @endcode
/// @par References
/// - Vezhnevets et al. (2017), FeUdal Networks for Hierarchical Reinforcement Learning.
///   https://proceedings.mlr.press/v70/vezhnevets17a.html
inline nn::Tensor feudal_direction(nn::ParameterStore& params,const nn::Tensor& states,
                                    std::int64_t hidden) {
  if (!states.defined() || states.dim()!=2 || states.size(1)<1 ||
      !nn::is_floating(states.dtype()) || hidden<1) {
    throw std::invalid_argument("feudal_direction: invalid states or width");
  }
  const auto features=nn::dense(params,"manager/hidden",states,hidden,nn::Activation::Tanh);
  return detail::feudal_unit(nn::dense(params,"manager/direction",features,states.size(1)));
}

/// @brief Score each action by the alignment of a learned vector with the goal.
/// @param params Worker parameter store or shared store with manager names.
/// @param states Floating state matrix [batch,coordinates].
/// @param goals Same-shaped direction rows of matching dtype and device. Detached during scoring.
/// @param hidden Positive hidden width.
/// @param actions Positive action count.
/// @return Intra-segment action logits [batch,actions].
/// @throws std::invalid_argument Invalid geometry or layer sizes.
/// @par Example
/// @code
/// dm::nn::ParameterStore params(1);
/// const auto x=dm::nn::eye(2);
/// std::print("{}\n",dm::rl::feudal_action_logits(params,x,x,16,4).size(1)); // 4
/// @endcode
/// @see feudal_direction()
inline nn::Tensor feudal_action_logits(nn::ParameterStore& params,const nn::Tensor& states,
    const nn::Tensor& goals,std::int64_t hidden,std::int64_t actions) {
  if (!states.defined() || states.dim()!=2 || states.size(1)<1 || !nn::is_floating(states.dtype()) ||
      !goals.defined() || goals.shape()!=states.shape() || goals.dtype()!=states.dtype() ||
      goals.device().backend!=states.device().backend || goals.device().index!=states.device().index ||
      hidden<1 || actions<1 || actions>std::numeric_limits<std::int64_t>::max()/states.size(1)) {
    throw std::invalid_argument("feudal_action_logits: invalid features or sizes");
  }
  const auto features=nn::dense(params,"worker/hidden",states,hidden,nn::Activation::Tanh);
  const auto vectors=nn::dense(params,"worker/action_vectors",features,actions*states.size(1))
      .reshape({states.size(0),actions,states.size(1)});
  return nn::sum(vectors*goals.detach().reshape({states.size(0),1,states.size(1)}),{2});
}

/// @brief Compute alignment between observed motion and a directional goal.
/// @param displacement Floating rows [batch,coordinates] of observed motion.
/// @param goals Same-shaped direction rows of matching dtype and device.
/// @return Alignment [batch]. Zero displacement has alignment zero.
/// @throws std::invalid_argument Incompatible tensors.
/// @par Example
/// @code
/// const auto motion=dm::nn::eye(2);
/// std::print("{:.0f}\n",dm::rl::feudal_alignment(motion,motion).sum().item()); // 2
/// @endcode
/// @see feudal_direction()
inline nn::Tensor feudal_alignment(const nn::Tensor& displacement,const nn::Tensor& goals) {
  if (!displacement.defined() || displacement.dim()!=2 || displacement.size(1)<1 ||
      !nn::is_floating(displacement.dtype()) || !goals.defined() ||
      goals.shape()!=displacement.shape() || goals.dtype()!=displacement.dtype() ||
      goals.device().backend!=displacement.device().backend ||
      goals.device().index!=displacement.device().index) {
    throw std::invalid_argument("feudal_alignment: incompatible rows");
  }
  return nn::sum(detail::feudal_unit(displacement)*goals,{1});
}

} // namespace datamunge::rl
