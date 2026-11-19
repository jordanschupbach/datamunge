/// @file
/// @brief Differentiable finite-horizon occupancy under known dynamics.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Propagate a finite-state policy and average its discounted visits.
///
/// This expansion is differentiable with respect to the policy, initial
/// distribution and dynamics. Transition rows and policy rows are assumed
/// to be normalized probabilities. Include absorbing transition rows for
/// terminal states that should retain probability mass.
/// @param policy Floating action probabilities [states,actions].
/// @param dynamics Floating successor probabilities [states*actions,states], state-major order.
/// @param initial Floating initial state probabilities [states].
/// @param horizon Positive number of transitions.
/// @param discount Discount in [0,1], default 0.95.
/// @return Normalized discounted state-action occupancy [states,actions]
/// and state probabilities [states] after the last transition.
/// @throws std::invalid_argument Invalid geometry, dtype or planning horizon.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto [visits,final]=dm::rl::finite_policy_occupancy(n::ones({2,1}),
///     n::from_vector<float>({0,1,0,1},{2,2}),n::from_vector<float>({1,0}),1);
/// std::print("{:.0f}\n",final.to_vector<float>()[1]); // 1
/// @endcode
inline std::pair<nn::Tensor,nn::Tensor> finite_policy_occupancy(const nn::Tensor& policy,
    const nn::Tensor& dynamics,const nn::Tensor& initial,std::int64_t horizon,double discount=.95) {
  namespace n=nn;
  if (!policy.defined() || !dynamics.defined() || !initial.defined() ||
      policy.dim()!=2 || policy.numel()==0 || !n::is_floating(policy.dtype()) ||
      dynamics.dtype()!=policy.dtype() || initial.dtype()!=policy.dtype() ||
      horizon<1 || !(discount>=0 && discount<=1)) {
    throw std::invalid_argument("finite_policy_occupancy: invalid inputs");
  }
  const auto states=policy.size(0),actions=policy.size(1);
  if (states>std::numeric_limits<std::int64_t>::max()/actions ||
      dynamics.shape()!=n::Shape{states*actions,states} || initial.shape()!=n::Shape{states}) {
    throw std::invalid_argument("finite_policy_occupancy: incompatible shapes");
  }
  for (const auto* field:{&dynamics,&initial}) {
    if (field->device().backend!=policy.device().backend || field->device().index!=policy.device().index) {
      throw std::invalid_argument("finite_policy_occupancy: mismatched devices");
    }
  }
  auto state=initial.reshape({1,states});
  auto visits=n::zeros_like(policy);
  double weight=1,total=0;
  for (std::int64_t step=0;step<horizon;++step) {
    const auto occupancy=state.reshape({states,1})*policy;
    visits=visits+weight*occupancy;
    state=n::matmul(occupancy.reshape({1,states*actions}),dynamics);
    total+=weight;
    weight*=discount;
  }
  return {visits/total,state.reshape({states})};
}

/// @brief Compute the exact-occupancy GAIL objectives from policy and discriminator inputs.
///
/// The discriminator distinguishes expert from learner visits. Its
/// gradient treats occupancies as constants. The actor minimizes
/// expected negative discriminator reward plus a weighted log-policy
/// term and treats the discriminator reward as fixed.
/// @param learner Differentiable learner occupancy [states,actions].
/// @param expert Expert occupancy, same shape, dtype and device.
/// @param log_policy Learner log probabilities, same shape, dtype and device.
/// @param discriminator Logits for expert membership, same shape, dtype and device.
/// @param entropy_weight Finite nonnegative entropy regularizer, default 0.01.
/// @return Discriminator loss, actor loss, and per-action reward.
/// @throws std::invalid_argument Invalid geometry, dtype, device or entropy weight.
/// @par Example
/// @code
/// const auto half=.5*dm::nn::ones({1,2});
/// const auto out=dm::rl::gail_occupancy_losses(half,half,dm::nn::log(half),dm::nn::zeros({1,2}));
/// std::print("{:.3f}\n",out[0].item()); // 0.693
/// @endcode
/// @par References
/// - Ho and Ermon (2016), Generative Adversarial Imitation Learning.
///   https://arxiv.org/abs/1606.03476
/// @see finite_policy_occupancy()
inline std::array<nn::Tensor,3> gail_occupancy_losses(const nn::Tensor& learner,
    const nn::Tensor& expert,const nn::Tensor& log_policy,const nn::Tensor& discriminator,
    double entropy_weight=.01) {
  namespace n=nn;
  if (!learner.defined() || learner.dim()!=2 || learner.numel()==0 ||
      !n::is_floating(learner.dtype()) || !(entropy_weight>=0 && std::isfinite(entropy_weight))) {
    throw std::invalid_argument("gail_occupancy_losses: invalid occupancy or entropy weight");
  }
  for (const auto* value:{&expert,&log_policy,&discriminator}) {
    if (!value->defined() || value->shape()!=learner.shape() || value->dtype()!=learner.dtype() ||
        value->device().backend!=learner.device().backend || value->device().index!=learner.device().index) {
      throw std::invalid_argument("gail_occupancy_losses: incompatible inputs");
    }
  }
  const auto reward=n::softplus(discriminator);
  const auto disc=.5*n::sum(expert.detach()*n::softplus(-discriminator)+learner.detach()*reward);
  const auto actor=n::sum(learner*(entropy_weight*log_policy-reward.detach()));
  return {disc,actor,reward};
}

/// @brief Compute AIRL objectives for deterministic dynamics and a state-only reward.
///
/// The shaped score adds discounted successor potential and subtracts
/// current potential. Discriminator logits subtract detached log-policy
/// probabilities. The actor treats the entire shaped score as fixed.
/// @param learner Learner discounted occupancy [states,actions].
/// @param expert Expert occupancy, same shape, dtype and device.
/// @param log_policy Learner log probabilities, same shape, dtype and device.
/// @param reward State reward [states], matching floating dtype and CPU storage.
/// @param potential State potential [states], matching reward.
/// @param successor CPU Int32 or Int64 successor indices [states,actions].
/// @param discount Finite shaping discount in [0,1], default 0.95.
/// @return Discriminator loss, actor loss, discriminator logits and shaped scores.
/// @throws std::invalid_argument Invalid tensors, indices or discount.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto out=dm::rl::airl_occupancy_losses(n::ones({1,1}),n::ones({1,1}),
///     n::zeros({1,1}),n::zeros({1}),n::zeros({1}),n::from_vector<int>({0},{1,1}));
/// std::print("{:.3f}\n",out[0].item()); // 0.693
/// @endcode
/// @par References
/// - Fu, Luo and Levine (2018), Learning Robust Rewards with Adversarial Inverse Reinforcement Learning.
///   https://arxiv.org/abs/1710.11248
/// @see gail_occupancy_losses()
inline std::array<nn::Tensor,4> airl_occupancy_losses(const nn::Tensor& learner,
    const nn::Tensor& expert,const nn::Tensor& log_policy,const nn::Tensor& reward,
    const nn::Tensor& potential,const nn::Tensor& successor,double discount=.95) {
  namespace n=nn;
  for (const auto* v:{&learner,&expert,&log_policy,&reward,&potential}) {
    if (!v->defined() || v->dtype()!=learner.dtype() || v->device().backend!=compute::Backend::CPU ||
        !n::is_floating(v->dtype())) {
      throw std::invalid_argument("airl_occupancy_losses: expected matching floating CPU tensors");
    }
  }
  if (learner.dim()!=2 || learner.numel()==0 || expert.shape()!=learner.shape() ||
      log_policy.shape()!=learner.shape() || reward.shape()!=n::Shape{learner.size(0)} ||
      potential.shape()!=reward.shape() || !successor.defined() || successor.shape()!=learner.shape() ||
      successor.device().backend!=compute::Backend::CPU ||
      (successor.dtype()!=n::DType::Int32 && successor.dtype()!=n::DType::Int64) ||
      !(discount>=0 && discount<=1)) {
    throw std::invalid_argument("airl_occupancy_losses: incompatible inputs");
  }
  for (const auto index:successor.to_vector<std::int64_t>()) {
    if (index<0 || index>=learner.size(0)) throw std::invalid_argument("AIRL successor out of range");
  }
  const auto shaping=discount*n::gather(potential,successor)-potential.reshape({-1,1});
  const auto score=reward.reshape({-1,1})+shaping;
  const auto logits=score-log_policy.detach();
  const auto disc=.5*n::sum(expert.detach()*n::softplus(-logits)+learner.detach()*n::softplus(logits));
  const auto actor=n::sum(learner*(log_policy-score.detach()));
  return {disc,actor,logits,score};
}

} // namespace datamunge::rl
