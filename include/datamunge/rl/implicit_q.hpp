/// @file
/// @brief Empirical expectile values and advantage-weighted cloning.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::rl {

/// @brief Fit a weighted expectile and an advantage-weighted policy in each state.
///
/// Only observed actions contribute to the expectile or actor. The
/// expectile minimizes asymmetric squared error exactly by considering
/// each interval between sorted action values. Empty rows yield zeros.
/// @param values Finite Float32 CPU action values [states,actions].
/// @param counts Finite nonnegative Float32 CPU empirical counts of the same shape.
/// @param tau Expectile in (0,1), default 0.7.
/// @param beta Finite nonnegative advantage multiplier, default 3.
/// @param cap Finite positive maximum exponential weight, default 100.
/// @return Float32 expectile values [states], policy [states,actions] and unnormalized actor weights.
/// @throws std::invalid_argument Invalid geometry, counts, values or options.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto out=dm::rl::implicit_q_policy(n::from_vector<float>({3,1},{1,2}),
///     n::from_vector<float>({0,1},{1,2}));
/// std::print("{:.0f}\n",out[0].item()); // 1
/// @endcode
/// @par References
/// - Kostrikov et al. (2022), Offline Reinforcement Learning with Implicit Q-Learning.
///   https://arxiv.org/abs/2110.06169
inline std::array<nn::Tensor,3> implicit_q_policy(const nn::Tensor& values,
    const nn::Tensor& counts,float tau=.7f,float beta=3.f,float cap=100.f) {
  namespace n=nn;
  if (!values.defined() || !counts.defined() || values.dim()!=2 || values.numel()==0 ||
      counts.shape()!=values.shape() || values.dtype()!=n::DType::Float32 ||
      counts.dtype()!=values.dtype() || values.device().backend!=compute::Backend::CPU ||
      counts.device().backend!=compute::Backend::CPU || !(tau>0 && tau<1) ||
      !(beta>=0 && std::isfinite(beta)) || !(cap>0 && std::isfinite(cap))) {
    throw std::invalid_argument("implicit_q_policy: invalid tables or options");
  }
  const auto q=values.to_vector<float>(), frequency=counts.to_vector<float>();
  const auto states=values.size(0),actions=values.size(1);
  std::vector<float> expectile(states), weights(q.size()), policy(q.size());
  for (std::int64_t state=0;state<states;++state) {
    const auto base=state*actions;
    std::vector<std::pair<float,float>> observed;
    for (std::int64_t action=0;action<actions;++action) {
      const auto i=base+action;
      if (!std::isfinite(q[i]) || !(frequency[i]>=0 && std::isfinite(frequency[i]))) {
        throw std::invalid_argument("implicit_q_policy: invalid value or count");
      }
      if (frequency[i]>0) observed.emplace_back(q[i],frequency[i]);
    }
    if (observed.empty()) continue;
    std::sort(observed.begin(),observed.end());
    std::vector<std::pair<double,double>> suffix(observed.size()+1);
    for (std::size_t i=observed.size();i>0;--i) {
      const auto [value,count]=observed[i-1];
      suffix[i-1]={suffix[i].first+count,suffix[i].second+double(value)*count};
    }
    double lower_mass=0,lower_sum=0,root=observed.back().first;
    for (std::size_t i=0;i<observed.size();++i) {
      const auto [value,count]=observed[i];
      const auto [upper_mass,upper_sum]=suffix[i];
      const double candidate=((1.-tau)*lower_sum+tau*upper_sum)/
          ((1.-tau)*lower_mass+tau*upper_mass);
      if (candidate<=value) {
        root=candidate;
        break;
      }
      lower_mass+=count;
      lower_sum+=double(value)*count;
    }
    expectile[state]=root;
    double total=0;
    for (std::int64_t action=0;action<actions;++action) {
      const auto i=base+action;
      if (frequency[i]==0) continue;
      const double weight=frequency[i]*std::exp(std::min(double(beta)*(q[i]-root),std::log(double(cap))));
      if (weight>std::numeric_limits<float>::max()) {
        throw std::invalid_argument("implicit_q_policy: actor weight exceeds Float32 range");
      }
      weights[i]=weight;
      total+=weights[i];
    }
    for (std::int64_t action=0;action<actions;++action) {
      const auto i=base+action;
      policy[i]=weights[i]/std::max(total,1e-12);
    }
  }
  return {n::from_vector(expectile),n::from_vector(policy,values.shape()),
          n::from_vector(weights,values.shape())};
}

} // namespace datamunge::rl
