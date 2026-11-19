/// @file
/// @brief Prior-regularized planning and occupancy-weighted distillation.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Plan one round of Distral and update the shared action prior.
///
/// Each task uses the same deterministic successor table. Soft Bellman
/// updates add a temperature-scaled log prior to action scores. The
/// distilled prior normalizes discounted, occupancy-weighted action
/// counts plus a pseudocount per action. Terminal values are zero.
/// @param successors CPU integer state indices [states,actions].
/// @param rewards Finite CPU Float32 rewards [tasks,states,actions].
/// @param live Binary CPU Float32 nonterminal flags [tasks,states].
/// @param initial CPU Float32 initial state probabilities [tasks,states], with no terminal mass.
/// @param prior Strictly positive CPU Float32 action probabilities [states,actions].
/// @param discount Discount in [0,1), default 0.9.
/// @param temperature Positive finite soft Bellman temperature, default 0.2.
/// @param prior_weight Nonnegative finite multiplier on log prior, default 0.8.
/// @param planning_steps Positive Bellman iteration count, default 100.
/// @param occupancy_steps Positive discounted occupancy horizon, default 160.
/// @param pseudocount Finite positive mass per action, default 0.001.
/// @return Updated values [tasks,states], task policies [tasks,states,actions],
/// occupancy [tasks,states], and distilled prior [states,actions].
/// @throws std::invalid_argument Incompatible tensors, invalid probabilities or planning options.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto [value,policy,visits,prior]=dm::rl::distral_round(
///     n::from_vector<int>({1,1},{2,1}),n::from_vector<float>({1,0},{1,2,1}),
///     n::from_vector<float>({1,0},{1,2}),n::from_vector<float>({1,0},{1,2}),n::ones({2,1}));
/// std::print("{:.0f}\n",value.to_vector<float>()[0]); // 1
/// @endcode
/// @par References
/// - Teh et al. (2017), Distral: Robust Multitask Reinforcement Learning.
///   https://arxiv.org/abs/1707.04175
inline std::array<nn::Tensor,4> distral_round(const nn::Tensor& successors,
    const nn::Tensor& rewards,const nn::Tensor& live,const nn::Tensor& initial,
    const nn::Tensor& prior,float discount=.9f,float temperature=.2f,float prior_weight=.8f,
    int planning_steps=100,int occupancy_steps=160,double pseudocount=.001) {
  namespace n=nn;
  for (const auto* field:{&rewards,&live,&initial,&prior}) {
    if (!field->defined() || field->dtype()!=n::DType::Float32 ||
        field->device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("distral_round: expected Float32 CPU data");
    }
  }
  if (rewards.dim()!=3 || rewards.numel()==0 || !successors.defined() ||
      successors.device().backend!=compute::Backend::CPU ||
      (successors.dtype()!=n::DType::Int64 && successors.dtype()!=n::DType::Int32) ||
      !(discount>=0 && discount<1) || !(temperature>0 && std::isfinite(temperature)) ||
      !(prior_weight>=0 && std::isfinite(prior_weight)) || planning_steps<1 ||
      occupancy_steps<1 || !(pseudocount>0 && std::isfinite(pseudocount))) {
    throw std::invalid_argument("distral_round: invalid transitions or planning options");
  }
  const auto tasks=rewards.size(0),states=rewards.size(1),actions=rewards.size(2);
  if (successors.shape()!=n::Shape{states,actions} || prior.shape()!=successors.shape() ||
      live.shape()!=n::Shape{tasks,states} || initial.shape()!=live.shape()) {
    throw std::invalid_argument("distral_round: incompatible shapes");
  }
  const auto next=successors.to_vector<std::int64_t>();
  const auto flags=live.to_vector<float>(),start=initial.to_vector<float>(),
             previous_prior=prior.to_vector<float>(),reward_data=rewards.to_vector<float>();
  if (!std::all_of(next.begin(),next.end(),[&](auto id){return id>=0 && id<states;}) ||
      !std::all_of(previous_prior.begin(),previous_prior.end(),[](float p){return p>0 && std::isfinite(p);}) ||
      !std::all_of(reward_data.begin(),reward_data.end(),[](float r){return std::isfinite(r);})) {
    throw std::invalid_argument("distral_round: invalid successor, prior or reward");
  }
  std::vector<std::int64_t> flat_next;
  std::vector<float> continuation;
  flat_next.reserve(rewards.numel());
  continuation.reserve(rewards.numel());
  for (std::int64_t task=0;task<tasks;++task) {
    const auto base=task*states;
    double mass=0;
    for (std::int64_t state=0;state<states;++state) {
      const float flag=flags[base+state], probability=start[base+state];
      if ((flag!=0 && flag!=1) || !(probability>=0 && std::isfinite(probability)) ||
          (flag==0 && probability!=0)) {
        throw std::invalid_argument("distral_round: invalid initial mass or terminal flag");
      }
      mass+=probability;
      for (std::int64_t action=0;action<actions;++action) {
        const auto index=base+next[state*actions+action];
        flat_next.push_back(index);
        continuation.push_back(flag*flags[index]);
      }
    }
    if (std::abs(mass-1)>1e-5) throw std::invalid_argument("distral_round: initial row must sum to one");
  }
  for (std::int64_t state=0;state<states;++state) {
    double mass=0;
    for (std::int64_t action=0;action<actions;++action) mass+=previous_prior[state*actions+action];
    if (std::abs(mass-1)>1e-5) throw std::invalid_argument("distral_round: prior row must sum to one");
  }
  const n::NoGradGuard guard;
  const auto indices=n::from_vector(flat_next,rewards.shape());
  const auto multiplier=n::from_vector(continuation,rewards.shape());
  const auto bias=prior_weight*n::log(prior);
  auto value=n::zeros({tasks,states});
  n::Tensor policy;
  for (int step=0;step<planning_steps;++step) {
    const auto q=rewards+discount*(multiplier*n::gather(value.reshape({-1}),indices));
    const auto scores=q/temperature+bias;
    policy=n::softmax(scores,-1);
    value=live*(temperature*n::logsumexp(scores,{2}));
  }
  const auto probabilities=policy.to_vector<float>();
  std::vector<float> occupancy(start.size());
  for (std::int64_t task=0;task<tasks;++task) {
    const auto base=task*states;
    std::vector<double> current(start.begin()+base,start.begin()+base+states);
    for (int step=0;step<occupancy_steps;++step) {
      std::vector<double> following(states);
      for (std::int64_t state=0;state<states;++state) {
        occupancy[base+state]+=current[state];
        for (std::int64_t action=0;action<actions;++action) {
          const auto index=(base+state)*actions+action;
          const auto successor=next[state*actions+action];
          if (continuation[index]) {
            following[successor]+=discount*current[state]*probabilities[index];
          }
        }
      }
      current=std::move(following);
    }
  }
  std::vector<float> distilled(previous_prior.size());
  for (std::int64_t state=0;state<states;++state) {
    std::vector<double> counts(actions,pseudocount);
    for (std::int64_t task=0;task<tasks;++task) {
      const auto row=task*states+state;
      for (std::int64_t action=0;action<actions;++action) {
        counts[action]+=occupancy[row]*probabilities[row*actions+action];
      }
    }
    double total=0;
    for (const double count:counts) total+=count;
    for (std::int64_t action=0;action<actions;++action) distilled[state*actions+action]=counts[action]/total;
  }
  return {value,policy,n::from_vector(occupancy,live.shape()),n::from_vector(distilled,prior.shape())};
}

} // namespace datamunge::rl
