/// @file
/// @brief Differentiable finite-horizon trajectory partition functions.
#pragma once

#include <cstdint>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Compute the maximum-entropy trajectory log partition for every start.
///
/// Backward dynamic programming sums unnormalized trajectory mass in
/// log space. Every action sequence has fixed length, so absorbing
/// states still contribute their outgoing action choices. Gradients
/// through the result give expected state-action visit counts.
/// @param rewards Floating CPU immediate rewards [states,actions].
/// @param successors CPU Int32 or Int64 successor IDs of the same shape.
/// @param horizon Nonnegative number of actions in each trajectory.
/// @return Log partition [states], matching the reward dtype.
/// @throws std::invalid_argument Invalid geometry, IDs or horizon.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto z=dm::rl::trajectory_log_partition(n::zeros({1,2}),
///     n::from_vector<int>({0,0},{1,2}),2);
/// std::print("{:.6f}\n",z.item()); // 1.386294
/// @endcode
/// @par References
/// - Ziebart et al. (2008), Maximum Entropy Inverse Reinforcement Learning.
///   https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf
inline nn::Tensor trajectory_log_partition(const nn::Tensor& rewards,
    const nn::Tensor& successors,std::int64_t horizon) {
  namespace n=nn;
  if (!rewards.defined() || !successors.defined() || rewards.dim()!=2 ||
      rewards.numel()==0 || !n::is_floating(rewards.dtype()) ||
      successors.shape()!=rewards.shape() ||
      rewards.device().backend!=compute::Backend::CPU || successors.device().backend!=compute::Backend::CPU ||
      (successors.dtype()!=n::DType::Int32 && successors.dtype()!=n::DType::Int64) || horizon<0) {
    throw std::invalid_argument("trajectory_log_partition: invalid tables or horizon");
  }
  for (const auto state:successors.to_vector<std::int64_t>()) {
    if (state<0 || state>=rewards.size(0)) {
      throw std::invalid_argument("trajectory_log_partition: successor out of range");
    }
  }
  auto log_mass=n::zeros({rewards.size(0)},rewards.dtype());
  for (std::int64_t step=0;step<horizon;++step) {
    log_mass=n::logsumexp(rewards+n::gather(log_mass,successors),{1});
  }
  return log_mass;
}

} // namespace datamunge::rl
