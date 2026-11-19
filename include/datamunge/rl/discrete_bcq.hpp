/// @file
/// @brief Count-based batch-constrained Q updates for discrete actions.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::rl {

/// @brief Apply a synchronous discrete BCQ sweep to a fixed empirical batch.
///
/// Candidate actions need positive counts and a relative behavior probability
/// at least as large as the threshold. Constrained selection and bootstrapping
/// use this support. Updates apply to all observed actions. Empty rows
/// bootstrap zero. In constrained mode they return action zero as a placeholder.
/// @param values Finite CPU Float32 Q-table [states,actions].
/// @param counts Nonnegative CPU Int32 or Int64 counts of the same shape.
/// @param successors CPU Int32 or Int64 successor IDs of the same shape.
/// @param rewards Finite CPU Float32 immediate rewards of the same shape.
/// @param live CPU Float32 binary bootstrap flag for each transition.
/// @param threshold Relative behavior likelihood in [0,1], default 0.3.
/// @param discount Discount in [0,1], default 0.95.
/// @param constrained Whether to mask actions when ranking and bootstrapping, default true.
/// @return Updated Q-table, Int64 greedy actions [states], Float32 behavior probabilities,
/// Int64 support mask, and Float32 TD targets. Ties use the lowest action index.
/// @throws std::invalid_argument Invalid geometry, values, indices or options.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto shape=n::Shape{1,2};
/// const auto out=dm::rl::discrete_bcq_sweep(n::from_vector<float>({3,0},shape),
///     n::from_vector<int>({0,1},shape),n::from_vector<int>({0,0},shape),
///     n::ones(shape),n::zeros(shape));
/// std::print("{}\n",out[1].to_vector<int>()[0]); // 1
/// @endcode
/// @par References
/// - Fujimoto et al. (2019), Benchmarking Batch Deep Reinforcement Learning Algorithms.
///   https://arxiv.org/abs/1910.01708
inline std::array<nn::Tensor,5> discrete_bcq_sweep(const nn::Tensor& values,
    const nn::Tensor& counts,const nn::Tensor& successors,const nn::Tensor& rewards,
    const nn::Tensor& live,float threshold=.3f,float discount=.95f,bool constrained=true) {
  namespace n=nn;
  if (!values.defined() || values.dim()!=2 || values.numel()==0 ||
      !(threshold>=0 && threshold<=1) || !(discount>=0 && discount<=1)) {
    throw std::invalid_argument("discrete_bcq_sweep: invalid geometry or options");
  }
  for (const auto* field:{&values,&counts,&successors,&rewards,&live}) {
    if (!field->defined() || field->shape()!=values.shape() ||
        field->device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("discrete_bcq_sweep: incompatible CPU table");
    }
  }
  for (const auto* field:{&counts,&successors}) {
    if (field->dtype()!=n::DType::Int32 && field->dtype()!=n::DType::Int64) {
      throw std::invalid_argument("discrete_bcq_sweep: expected integer counts and indices");
    }
  }
  for (const auto* field:{&values,&rewards,&live}) {
    if (field->dtype()!=n::DType::Float32) {
      throw std::invalid_argument("discrete_bcq_sweep: expected Float32 values and rewards");
    }
  }
  const auto count=counts.to_vector<std::int64_t>(), next=successors.to_vector<std::int64_t>();
  const auto q=values.to_vector<float>(), reward=rewards.to_vector<float>(), active=live.to_vector<float>();
  const auto states=values.size(0),actions=values.size(1);
  for (std::size_t i=0;i<q.size();++i) {
    if (!std::isfinite(q[i]) || !std::isfinite(reward[i]) || count[i]<0 ||
        next[i]<0 || next[i]>=states || (active[i]!=0.f && active[i]!=1.f)) {
      throw std::invalid_argument("discrete_bcq_sweep: invalid batch values");
    }
  }
  std::vector<float> probability(q.size()), maximum(states), td(q.size());
  std::vector<std::int64_t> support(q.size()), policy(states);
  for (std::int64_t state=0;state<states;++state) {
    const auto base=state*actions;
    double total=0;
    for (std::int64_t a=0;a<actions;++a) total+=count[base+a];
    if (total==0) {
      if (!constrained) {
        policy[state]=std::max_element(q.begin()+base,q.begin()+base+actions)-(q.begin()+base);
      }
      continue;
    }
    float peak=0;
    for (std::int64_t a=0;a<actions;++a) {
      probability[base+a]=count[base+a]/total;
      peak=std::max(peak,probability[base+a]);
    }
    std::int64_t best=-1;
    for (std::int64_t a=0;a<actions;++a) {
      const auto i=base+a;
      support[i]=count[i]>0 && probability[i]>=threshold*peak;
      if ((!constrained || support[i]) && (best<0 || q[i]>q[base+best])) best=a;
    }
    policy[state]=best;
    maximum[state]=q[base+best];
  }
  auto updated=q;
  for (std::size_t i=0;i<q.size();++i) {
    td[i]=reward[i]+discount*(active[i]*maximum[next[i]]);
    if (count[i]>0) updated[i]=td[i];
  }
  return {n::from_vector(updated,values.shape()),n::from_vector(policy),
      n::from_vector(probability,values.shape()),n::from_vector(support,values.shape()),
      n::from_vector(td,values.shape())};
}

} // namespace datamunge::rl
